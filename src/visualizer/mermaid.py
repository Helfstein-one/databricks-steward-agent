"""Mermaid diagram generator for ER diagrams (Crow's foot) and Medallion lineage flowcharts."""

from __future__ import annotations

import re

from src.semantic.models import EntityModel, RelationshipModel


def _sanitize_id(name: str) -> str:
    """Sanitize identifier for Mermaid syntax."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _sanitize_type(data_type: str) -> str:
    """Normalize data type to valid single-word Mermaid attribute type."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", data_type.lower())
    return cleaned if cleaned else "string"


def generate_er_diagram(
    entities: list[EntityModel],
    relationships: list[RelationshipModel] | None = None,
) -> str:
    """Generate Mermaid erDiagram with Crow's foot cardinality notation.

    Args:
        entities: List of EntityModel objects to include.
        relationships: Optional explicit list of relationships. If None, gathers from entities.

    Returns:
        String containing formatted Mermaid erDiagram code block.
    """
    lines: list[str] = ["erDiagram"]

    # Collect all relationships
    all_rels: list[RelationshipModel] = []
    if relationships is not None:
        all_rels.extend(relationships)
    else:
        for ent in entities:
            all_rels.extend(ent.relationships)

    # Cardinality mapping
    cardinality_map = {
        "many_to_one": "}o--||",
        "one_to_many": "||--o{",
        "one_to_one": "||--||",
        "many_to_many": "}o--o{",
    }

    # Emit relationships first
    seen_rels = set()
    for rel in all_rels:
        from_id = _sanitize_id(rel.from_entity)
        to_id = _sanitize_id(rel.to_entity)
        rel_type = rel.type.lower()
        symbol = cardinality_map.get(rel_type, "}o--||")
        label = rel.description or f"{rel.from_column}_to_{rel.to_column}"
        label_clean = re.sub(r'["\n]', "", label)

        rel_key = (from_id, to_id, rel.from_column, rel.to_column)
        if rel_key not in seen_rels:
            lines.append(f'    {from_id} {symbol} {to_id} : "{label_clean}"')
            seen_rels.add(rel_key)

    # Foreign key lookup across relationships
    fk_map: dict[str, set[str]] = {}
    for rel in all_rels:
        from_ent = rel.from_entity.lower()
        if from_ent not in fk_map:
            fk_map[from_ent] = set()
        fk_map[from_ent].add(rel.from_column.lower())

    # Emit entity attribute definitions
    for ent in entities:
        ent_id = _sanitize_id(ent.name)
        lines.append(f"    {ent_id} {{")

        ent_fks = fk_map.get(ent.name.lower(), set())

        # If columns defined
        if ent.columns:
            for col in ent.columns:
                col_name = _sanitize_id(col.name)
                col_type = _sanitize_type(col.type)
                key_type = ""
                if col.primary_key or (ent.primary_key and col.name == ent.primary_key):
                    key_type = " PK"
                elif col.name.lower() in ent_fks:
                    key_type = " FK"

                desc = f' "{col.description}"' if col.description else ""
                lines.append(f"        {col_type} {col_name}{key_type}{desc}")
        elif ent.dimensions:
            for dim in ent.dimensions:
                col_name = _sanitize_id(dim.column or dim.name)
                col_type = _sanitize_type(dim.type)
                key_type = ""
                if ent.primary_key and (dim.name == ent.primary_key or dim.column == ent.primary_key):
                    key_type = " PK"
                elif col_name.lower() in ent_fks:
                    key_type = " FK"
                lines.append(f"        {col_type} {col_name}{key_type}")
        else:
            lines.append("        string id PK")

        lines.append("    }")

    return "\n".join(lines)


def generate_lineage_diagram(entities: list[EntityModel]) -> str:
    """Generate Mermaid flowchart (graph LR) partitioned into Medallion layers.

    Args:
        entities: List of EntityModel objects.

    Returns:
        String containing formatted Mermaid graph LR code block.
    """
    lines: list[str] = ["graph LR"]

    bronze_nodes: list[str] = []
    silver_nodes: list[str] = []
    gold_nodes: list[str] = []
    other_nodes: list[str] = []

    for ent in entities:
        ent_id = _sanitize_id(ent.name)
        display_name = ent.name.replace("_", " ").title()
        node_def = f'        {ent_id}["{display_name}<br/><i>{ent.table_name or ent.name}</i>"]'

        layer = (ent.layer or "").lower()
        if not layer:
            name_lower = ent.name.lower()
            if "bronze" in name_lower or "raw" in name_lower:
                layer = "bronze"
            elif "silver" in name_lower or "clean" in name_lower:
                layer = "silver"
            elif "gold" in name_lower or "kpi" in name_lower or "agg" in name_lower:
                layer = "gold"

        if layer == "bronze":
            bronze_nodes.append(node_def)
        elif layer == "silver":
            silver_nodes.append(node_def)
        elif layer == "gold":
            gold_nodes.append(node_def)
        else:
            other_nodes.append(node_def)

    # Subgraphs for Medallion
    if bronze_nodes:
        lines.append("    subgraph Bronze [Bronze Layer: Raw Ingestion]")
        lines.extend(bronze_nodes)
        lines.append("    end")

    if silver_nodes:
        lines.append("    subgraph Silver [Silver Layer: Cleansed & Conformed]")
        lines.extend(silver_nodes)
        lines.append("    end")

    if gold_nodes:
        lines.append("    subgraph Gold [Gold Layer: Aggregated Business KPIs]")
        lines.extend(gold_nodes)
        lines.append("    end")

    if other_nodes:
        lines.append("    subgraph Entities [Lakehouse Entities]")
        lines.extend(other_nodes)
        lines.append("    end")

    # Connect nodes via relationships or default medallion flow
    edges_added = False
    for ent in entities:
        ent_id = _sanitize_id(ent.name)
        for rel in ent.relationships:
            target_id = _sanitize_id(rel.to_entity)
            lines.append(f"    {ent_id} -->|{rel.from_column}| {target_id}")
            edges_added = True

    # If no explicit relationships between layers, synthesize default medallion flow
    if not edges_added:
        if bronze_nodes and silver_nodes:
            first_bronze = re.search(r"^\s+([a-zA-Z0-9_]+)\[", bronze_nodes[0])
            first_silver = re.search(r"^\s+([a-zA-Z0-9_]+)\[", silver_nodes[0])
            if first_bronze and first_silver:
                lines.append(f"    {first_bronze.group(1)} -->|ETL Dedup & Clean| {first_silver.group(1)}")

        if silver_nodes and gold_nodes:
            first_silver = re.search(r"^\s+([a-zA-Z0-9_]+)\[", silver_nodes[0])
            first_gold = re.search(r"^\s+([a-zA-Z0-9_]+)\[", gold_nodes[0])
            if first_silver and first_gold:
                lines.append(f"    {first_silver.group(1)} -->|Aggregate KPIs| {first_gold.group(1)}")

    return "\n".join(lines)
