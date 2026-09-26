"""Auto-discovery CLI crawler for Databricks Unity Catalog with LLM enrichment.

Extracts table schemas from Databricks Unity Catalog via DatabricksAdapter,
runs a LangChain prompt chain to enrich dimensions, metrics, and business descriptions,
and outputs the enriched YAML semantic model conforming to SemanticDomainModel.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.adapters.databricks_adapter import DatabricksAdapter
from src.config import settings
from src.semantic.models import (
    DimensionModel,
    EntityModel,
    MetricModel,
    SemanticDomainModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def enrich_entity_with_llm(entity: EntityModel, llm: ChatOpenAI | None = None) -> EntityModel:
    """Enrich an EntityModel with dimensions, metrics, and business descriptions using LLM or fallback rules."""
    cols = entity.columns or []
    col_summary = [
        {"name": c.name, "type": c.type, "nullable": c.nullable, "primary_key": c.primary_key}
        for c in cols
    ]

    enriched_data: dict[str, Any] | None = None

    if llm is not None:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert Data Steward and Semantic Layer Engineer.
Analyze the given table schema and generate an enriched semantic definition in valid JSON format.

JSON Schema required:
{{
    "description": "Business description for the entity",
    "synonyms": ["synonym1", "synonym2"],
    "dimensions": [
        {{
            "name": "column_name",
            "type": "string|date|timestamp|int|double|boolean",
            "column": "column_name",
            "description": "Business description of column",
            "synonyms": ["synonym1"]
        }}
    ],
    "metrics": [
        {{
            "name": "metric_name",
            "type": "sum|avg|count|count_distinct|ratio",
            "sql": "SQL_AGGREGATION_EXPRESSION",
            "description": "Business description of metric",
            "synonyms": ["synonym1"]
        }}
    ]
}}

Rules:
1. Every numeric column (int, bigint, float, double, decimal) should have candidate metrics like sum or avg, UNLESS it is an ID column.
2. The primary key or ID columns should have a count or count_distinct metric.
3. Write clear, high-quality business descriptions.
4. Output raw JSON ONLY with no markdown wrapping.
""",
                    ),
                    (
                        "user",
                        "Entity Name: {entity_name}\nTable: {table_name}\nColumns: {columns_json}",
                    ),
                ]
            )

            chain = prompt | llm
            res = chain.invoke(
                {
                    "entity_name": entity.name,
                    "table_name": entity.table_name or entity.name,
                    "columns_json": json.dumps(col_summary),
                }
            )
            raw_content = str(getattr(res, "content", "") or "").strip()
            # Extract JSON block
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                raw_content = match.group(0)

            enriched_data = json.loads(raw_content)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "LLM enrichment failed for entity '%s' (%s). Falling back to heuristic rule engine.",
                entity.name,
                err,
            )

    if not enriched_data:
        # Fallback heuristic enrichment
        dimensions: list[DimensionModel] = []
        metrics: list[MetricModel] = []

        pk = entity.primary_key
        if not pk:
            for c in cols:
                if c.name.lower().endswith("_id") or c.name.lower() in ("id", "key"):
                    pk = c.name
                    break

        if pk:
            metrics.append(
                MetricModel(
                    name=f"total_{entity.name}_count",
                    entity=entity.name,
                    type="count",
                    sql=f"COUNT(DISTINCT {pk})",
                    description=f"Total count of unique {pk} records in {entity.name}",
                    synonyms=[f"total_{entity.name}"],
                )
            )
        else:
            metrics.append(
                MetricModel(
                    name=f"total_{entity.name}_records",
                    entity=entity.name,
                    type="count",
                    sql="COUNT(*)",
                    description=f"Total record count for entity {entity.name}",
                )
            )

        for col in cols:
            dimensions.append(
                DimensionModel(
                    name=col.name,
                    type=col.type,
                    column=col.name,
                    description=col.description
                    or f"Dimension attribute {col.name} of {entity.name}",
                    synonyms=[col.name.replace("_", " ")],
                )
            )

            col_type = col.type.lower()
            if any(
                t in col_type for t in ("double", "float", "decimal", "numeric", "int", "bigint")
            ) and not (col.name.lower().endswith("_id") or col.name.lower() in ("id", "key")):
                metrics.append(
                    MetricModel(
                        name=f"sum_{col.name}",
                        entity=entity.name,
                        type="sum",
                        sql=f"SUM({col.name})",
                        description=f"Total sum of {col.name} in {entity.name}",
                        synonyms=[f"total_{col.name}"],
                    )
                )
                metrics.append(
                    MetricModel(
                        name=f"avg_{col.name}",
                        entity=entity.name,
                        type="avg",
                        sql=f"AVG({col.name})",
                        description=f"Average {col.name} in {entity.name}",
                        synonyms=[f"average_{col.name}"],
                    )
                )

        return EntityModel(
            name=entity.name,
            table_name=entity.table_name,
            catalog=entity.catalog,
            schema_name=entity.schema_name,
            primary_key=pk,
            layer=entity.layer,
            columns=cols,
            dimensions=dimensions,
            metrics=metrics,
            description=entity.description or f"Discovered entity {entity.name} from Unity Catalog",
            synonyms=[entity.name, entity.name.replace("_", " ")],
        )

    # Convert LLM dict to Pydantic objects
    dims = [
        DimensionModel(
            name=d.get("name", ""),
            type=d.get("type", "string"),
            column=d.get("column", d.get("name", "")),
            description=d.get("description"),
            synonyms=d.get("synonyms", []),
        )
        for d in enriched_data.get("dimensions", [])
    ]

    mets = [
        MetricModel(
            name=m.get("name", ""),
            entity=entity.name,
            type=m.get("type", "sum"),
            sql=m.get("sql", "COUNT(*)"),
            description=m.get("description"),
            synonyms=m.get("synonyms", []),
        )
        for m in enriched_data.get("metrics", [])
    ]

    return EntityModel(
        name=entity.name,
        table_name=entity.table_name,
        catalog=entity.catalog,
        schema_name=entity.schema_name,
        primary_key=entity.primary_key,
        layer=entity.layer,
        columns=cols,
        dimensions=dims,
        metrics=mets,
        description=enriched_data.get("description") or entity.description,
        synonyms=enriched_data.get("synonyms", [entity.name]),
    )


def run_discovery_crawler(
    catalog: str | None = None,
    schema: str | None = None,
    output_path: Path | str | None = None,
    use_llm: bool = True,
) -> SemanticDomainModel:
    """Execute the auto-discovery crawler on Unity Catalog and save enriched YAML model."""
    adapter = DatabricksAdapter()
    cat = catalog or settings.databricks_default_catalog or "workspace"
    sch = schema or settings.databricks_default_schema or "default"

    logger.info("Starting auto-discovery crawler for catalog='%s', schema='%s'", cat, sch)

    raw_entities = adapter.introspect_catalog(catalog=cat, schema=sch)
    logger.info("Discovered %d entities in catalog introspection.", len(raw_entities))

    llm_instance: ChatOpenAI | None = None
    if use_llm:
        try:
            llm_instance = ChatOpenAI(
                base_url=settings.local_llm_base_url,
                api_key=settings.local_llm_api_key,
                model=settings.local_llm_model,
                temperature=settings.local_llm_temperature,
                timeout=10.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.info("LLM setup unavailable (%s). Crawler will use offline enrichment.", e)

    enriched_entities: list[EntityModel] = []
    for raw_ent in raw_entities:
        logger.info("Enriching entity '%s'...", raw_ent.name)
        enriched = enrich_entity_with_llm(raw_ent, llm=llm_instance)
        enriched_entities.append(enriched)

    domain_name = f"auto_discovered_{sch}" if sch else "auto_discovered_catalog"
    domain_model = SemanticDomainModel(
        domain=domain_name,
        description=f"Auto-discovered semantic model for catalog `{cat}` and schema `{sch}`.",
        catalog=cat,
        schema_name=sch,
        entities=enriched_entities,
    )

    out_file = (
        Path(output_path)
        if output_path
        else (settings.semantic_models_path / f"{domain_name}.yaml")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert domain model to dict and write clean YAML
    dump_data = domain_model.model_dump(exclude_unset=True, exclude_none=True)

    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(dump_data, f, sort_keys=False, allow_unicode=True)

    logger.info("Successfully generated enriched YAML semantic model at: %s", out_file)
    return domain_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Databricks Unity Catalog LLM-Enriched Semantic Auto-Discovery Crawler"
    )
    parser.add_argument("--catalog", type=str, help="Unity Catalog name", default=None)
    parser.add_argument("--schema", type=str, help="Schema/database name", default=None)
    parser.add_argument("--output", type=str, help="Output YAML file path", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enrichment chain")

    args = parser.parse_args()
    run_discovery_crawler(
        catalog=args.catalog,
        schema=args.schema,
        output_path=args.output,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
