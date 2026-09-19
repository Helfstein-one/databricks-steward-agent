"""Semantic registry managing semantic models, synonyms, and prompt context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.semantic.models import (
    DimensionModel,
    EntityModel,
    MetricModel,
    RelationshipModel,
    SemanticDomainModel,
)


class SemanticRegistry:
    """Registry for loading, indexing, and querying semantic domain ontologies."""

    def __init__(self, models_dir: Path | str | None = None):
        self.models_dir = Path(models_dir) if models_dir else None
        self.domains: dict[str, SemanticDomainModel] = {}
        self.entities: dict[str, EntityModel] = {}
        self.metrics: dict[str, MetricModel] = {}
        self.relationships: list[RelationshipModel] = []
        self._synonym_index: dict[str, dict[str, Any]] = {}

        if models_dir:
            self.load_directory(models_dir)

    def load_directory(self, path: Path | str) -> None:
        """Load all YAML semantic models from a directory."""
        dir_path = Path(path)
        if not dir_path.exists():
            return

        for file_path in dir_path.glob("*.yaml"):
            self.load_file(file_path)

    def load_file(self, file_path: Path | str) -> SemanticDomainModel:
        """Load a single YAML semantic model file and register its contents."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Semantic model file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError(f"Invalid YAML content in {path}: expected a dictionary")

        model = SemanticDomainModel(**data)
        self.register_domain(model)
        return model

    def register_domain(self, domain_model: SemanticDomainModel) -> None:
        """Register a domain model and index its entities, metrics, and synonyms."""
        self.domains[domain_model.domain.lower()] = domain_model

        for entity in domain_model.entities:
            self.entities[entity.name.lower()] = entity

            # Index entity synonyms
            self._synonym_index[entity.name.lower()] = {
                "type": "entity",
                "entity": entity.name,
                "target": entity.name,
            }
            for syn in entity.synonyms:
                self._synonym_index[syn.lower()] = {
                    "type": "entity",
                    "entity": entity.name,
                    "target": entity.name,
                }

            # Index dimensions
            for dim in entity.dimensions:
                dim_key = f"{entity.name.lower()}.{dim.name.lower()}"
                self._synonym_index[dim_key] = {
                    "type": "dimension",
                    "entity": entity.name,
                    "target": dim.name,
                }
                for syn in dim.synonyms:
                    self._synonym_index[syn.lower()] = {
                        "type": "dimension",
                        "entity": entity.name,
                        "target": dim.name,
                    }

            # Index entity metrics
            for metric in entity.metrics:
                m_key = f"{entity.name.lower()}.{metric.name.lower()}"
                self.metrics[metric.name.lower()] = metric
                self._synonym_index[m_key] = {
                    "type": "metric",
                    "entity": entity.name,
                    "target": metric.name,
                }
                for syn in metric.synonyms:
                    self._synonym_index[syn.lower()] = {
                        "type": "metric",
                        "entity": entity.name,
                        "target": metric.name,
                    }

        # Index domain-level metrics
        for metric in domain_model.metrics:
            self.metrics[metric.name.lower()] = metric
            self._synonym_index[metric.name.lower()] = {
                "type": "metric",
                "entity": metric.entity,
                "target": metric.name,
            }
            for syn in metric.synonyms:
                self._synonym_index[syn.lower()] = {
                    "type": "metric",
                    "entity": metric.entity,
                    "target": metric.name,
                }

        # Register relationships from both entities and domain level
        existing_rel_keys = {
            (
                r.from_entity.lower(),
                r.from_column.lower(),
                r.to_entity.lower(),
                r.to_column.lower(),
            )
            for r in self.relationships
        }

        # 1. Collect entity-level relationships
        for entity in domain_model.entities:
            for rel in entity.relationships:
                key = (
                    rel.from_entity.lower(),
                    rel.from_column.lower(),
                    rel.to_entity.lower(),
                    rel.to_column.lower(),
                )
                if key not in existing_rel_keys:
                    self.relationships.append(rel)
                    existing_rel_keys.add(key)

        # 2. Collect domain-level relationships
        for rel in domain_model.relationships:
            key = (
                rel.from_entity.lower(),
                rel.from_column.lower(),
                rel.to_entity.lower(),
                rel.to_column.lower(),
            )
            if key not in existing_rel_keys:
                self.relationships.append(rel)
                existing_rel_keys.add(key)

    def get_domain(self, name: str) -> SemanticDomainModel | None:
        """Retrieve domain model by name."""
        return self.domains.get(name.lower())

    def get_entity(self, name: str) -> EntityModel | None:
        """Retrieve entity by name or synonym."""
        name_lower = name.lower()
        if name_lower in self.entities:
            return self.entities[name_lower]

        syn = self._synonym_index.get(name_lower)
        if syn and syn["type"] == "entity":
            return self.entities.get(syn["target"].lower())

        return None

    def get_metric(self, name: str, entity_name: str | None = None) -> MetricModel | None:
        """Retrieve metric by name or synonym, optionally scoped to an entity."""
        result = self.find_metric_owner(name, preferred_entity=entity_name)
        return result[0] if result else None

    def find_metric_owner(
        self, metric_name: str, preferred_entity: str | None = None
    ) -> tuple[MetricModel, str] | None:
        """Find metric and the entity that owns it."""
        m_lower = metric_name.lower()

        # Check synonym
        syn = self._synonym_index.get(m_lower)
        target_name = syn["target"].lower() if syn and syn["type"] == "metric" else m_lower

        if preferred_entity:
            entity = self.get_entity(preferred_entity)
            if entity:
                for m in entity.metrics:
                    if m.name.lower() == target_name or target_name in [
                        s.lower() for s in m.synonyms
                    ]:
                        return m, entity.name

        for entity in self.entities.values():
            for m in entity.metrics:
                if m.name.lower() == target_name or target_name in [s.lower() for s in m.synonyms]:
                    return m, entity.name

        if target_name in self.metrics:
            m = self.metrics[target_name]
            return m, m.entity or (preferred_entity or "")

        return None

    def get_dimension(
        self, dimension_name: str, entity_name: str | None = None
    ) -> DimensionModel | None:
        """Retrieve dimension by name or synonym."""
        result = self.find_dimension_owner(dimension_name, preferred_entity=entity_name)
        return result[0] if result else None

    def find_dimension_owner(
        self, dimension_name: str, preferred_entity: str | None = None
    ) -> tuple[DimensionModel, str] | None:
        """Find dimension and the entity that owns it."""
        dim_lower = dimension_name.lower()

        # Check synonym
        syn = self._synonym_index.get(dim_lower)
        target_name = syn["target"].lower() if syn and syn["type"] == "dimension" else dim_lower

        if preferred_entity:
            entity = self.get_entity(preferred_entity)
            if entity:
                for dim in entity.dimensions:
                    if (
                        dim.name.lower() == target_name
                        or dim.column.lower() == target_name
                        or target_name in [s.lower() for s in dim.synonyms]
                    ):
                        return dim, entity.name

        for entity in self.entities.values():
            for dim in entity.dimensions:
                if (
                    dim.name.lower() == target_name
                    or dim.column.lower() == target_name
                    or target_name in [s.lower() for s in dim.synonyms]
                ):
                    return dim, entity.name

        return None

    def resolve_synonym(self, synonym: str) -> dict[str, Any] | None:
        """Resolve a business synonym to its entity, dimension, or metric mapping."""
        return self._synonym_index.get(synonym.lower())

    def list_entities(self, domain_name: str | None = None) -> list[EntityModel]:
        """List all registered entities, optionally filtered by domain."""
        if domain_name:
            domain = self.get_domain(domain_name)
            return domain.entities if domain else []
        return list(self.entities.values())

    def list_metrics(self, entity_name: str | None = None) -> list[MetricModel]:
        """List all registered metrics, optionally filtered by entity."""
        if entity_name:
            entity = self.get_entity(entity_name)
            return entity.metrics if entity else []
        return list(self.metrics.values())

    def list_relationships(self, domain_name: str | None = None) -> list[RelationshipModel]:
        """List all registered relationships, optionally filtered by domain."""
        if domain_name:
            domain = self.get_domain(domain_name)
            if not domain:
                return []
            domain_rels: list[RelationshipModel] = []
            seen: set[tuple[str, str, str, str]] = set()

            for rel in domain.relationships:
                key = (
                    rel.from_entity.lower(),
                    rel.from_column.lower(),
                    rel.to_entity.lower(),
                    rel.to_column.lower(),
                )
                if key not in seen:
                    domain_rels.append(rel)
                    seen.add(key)

            for entity in domain.entities:
                for rel in entity.relationships:
                    key = (
                        rel.from_entity.lower(),
                        rel.from_column.lower(),
                        rel.to_entity.lower(),
                        rel.to_column.lower(),
                    )
                    if key not in seen:
                        domain_rels.append(rel)
                        seen.add(key)

            return domain_rels
        return self.relationships

    def get_prompt_context(self, domain_name: str | None = None) -> str:
        """Format semantic models into natural language string for system prompt injection."""
        if domain_name and self.get_domain(domain_name):
            domains_to_format = [self.get_domain(domain_name)]
        else:
            all_doms = list(self.domains.values())
            all_doms.sort(key=lambda d: 0 if d.domain == "databricks_medallion" else 1)
            domains_to_format = all_doms

        if not domains_to_format:
            return "No semantic models registered."

        sections: list[str] = []
        for domain in domains_to_format:
            if not domain:
                continue
            section = [f"### Domain: {domain.domain}"]
            if domain.description:
                section.append(f"Description: {domain.description}")

            section.append("\n**Entities & Tables:**")
            for entity in domain.entities:
                dims_str = ", ".join([f"{d.name} ({d.type})" for d in entity.dimensions])
                section.append(f"- Entity: `{entity.name}` (Table: `{entity.table_name}`)")
                if entity.primary_key:
                    section.append(f"  Primary Key: `{entity.primary_key}`")
                if dims_str:
                    section.append(f"  Dimensions: {dims_str}")
                if entity.metrics:
                    m_str = ", ".join([f"{m.name} [{m.sql}]" for m in entity.metrics])
                    section.append(f"  Metrics: {m_str}")

            rels_to_show = self.list_relationships(domain.domain)
            if rels_to_show:
                section.append("\n**Relationships (Joins):**")
                for rel in rels_to_show:
                    section.append(
                        f"- `{rel.from_entity}.{rel.from_column}` -> `{rel.to_entity}.{rel.to_column}` ({rel.type})"
                    )

            sections.append("\n".join(section))

        return "\n\n".join(sections)

    def register_data_product_entity(
        self,
        entity_name: str,
        table_name: str,
        columns: list[dict[str, Any]],
        source_entity: str | None = None,
        domain_name: str = "databricks_medallion",
        models_dir: Path | str | None = None,
    ) -> EntityModel | None:
        """Register a new data product entity into the YAML semantic model and reload the registry."""
        target_dir = (
            Path(models_dir) if models_dir else (self.models_dir or Path("configs/semantic_models"))
        )
        yaml_file = target_dir / f"{domain_name}.yaml"
        if not yaml_file.exists():
            yaml_files = list(target_dir.glob("*.yaml"))
            if yaml_files:
                yaml_file = yaml_files[0]

        raw_data: dict[str, Any] = {}
        if yaml_file.exists():
            with open(yaml_file, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}

        # 1. Infer primary key
        col_names = [c.get("name", "") for c in columns if isinstance(c, dict)]
        pk = next(
            (c for c in col_names if c.endswith("_id") or c == "id" or "key" in c),
            col_names[0] if col_names else "id",
        )

        # 2. Build dimensions
        dimensions: list[dict[str, Any]] = []
        for col in columns:
            c_name = col.get("name", "")
            c_type = str(col.get("type", "string")).lower()
            std_type = (
                "double"
                if any(t in c_type for t in ("double", "float", "decimal", "numeric"))
                else (
                    "bigint"
                    if any(t in c_type for t in ("int", "bigint", "long"))
                    else (
                        "timestamp"
                        if "timestamp" in c_type
                        else ("date" if "date" in c_type else "string")
                    )
                )
            )
            dimensions.append(
                {
                    "name": c_name,
                    "type": std_type,
                    "column": c_name,
                    "description": f"Coluna {c_name} do data product {entity_name}",
                }
            )

        # 3. Build metrics
        metrics: list[dict[str, Any]] = [
            {
                "name": f"total_{entity_name}_records",
                "type": "count",
                "sql": "COUNT(*)",
                "description": f"Contagem total de registros em {entity_name}",
            }
        ]
        for col in columns:
            c_name = col.get("name", "")
            c_type = str(col.get("type", "string")).lower()
            if any(
                t in c_type for t in ("double", "float", "decimal", "numeric", "int", "bigint")
            ) and not (c_name.endswith("_id") or c_name == "id"):
                metrics.append(
                    {
                        "name": f"total_{c_name}",
                        "type": "sum",
                        "sql": f"SUM({c_name})",
                        "description": f"Soma agregada de {c_name}",
                    }
                )
                metrics.append(
                    {
                        "name": f"avg_{c_name}",
                        "type": "avg",
                        "sql": f"AVG({c_name})",
                        "description": f"Média de {c_name}",
                    }
                )

        # 4. Build relationships
        relationships: list[dict[str, Any]] = []
        if source_entity and source_entity in self.entities:
            src = self.entities[source_entity]
            src_cols = [d.column for d in src.dimensions]
            common_col = next((c for c in col_names if c in src_cols), src.primary_key)
            if common_col:
                relationships.append(
                    {
                        "name": f"{source_entity}_to_{entity_name}",
                        "from_entity": source_entity,
                        "from_column": common_col,
                        "to_entity": entity_name,
                        "to_column": common_col,
                        "type": "many_to_one",
                    }
                )

        new_entity_dict: dict[str, Any] = {
            "name": entity_name,
            "table": table_name,
            "primary_key": pk,
            "layer": "gold" if "gold" in entity_name.lower() else "silver",
            "description": f"Data Product {entity_name} gerado e materializado no Databricks Unity Catalog.",
            "synonyms": [entity_name, entity_name.replace("_", " ")],
            "dimensions": dimensions,
            "metrics": metrics,
        }
        if relationships:
            new_entity_dict["relationships"] = relationships

        entities_list = raw_data.setdefault("entities", [])
        existing_idx = next(
            (i for i, e in enumerate(entities_list) if e.get("name") == entity_name), None
        )
        if existing_idx is not None:
            entities_list[existing_idx] = new_entity_dict
        else:
            entities_list.append(new_entity_dict)

        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(raw_data, f, sort_keys=False, allow_unicode=True)

        if target_dir.exists():
            self.load_directory(target_dir)

        return self.get_entity(entity_name)
