"""Declarative semantic layer models using Pydantic v2."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ColumnModel(BaseModel):
    """Physical or logical table column."""

    name: str
    type: str = "string"
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None
    description: str | None = None


class DimensionModel(BaseModel):
    """Business dimension mapping business concept to table column or expression."""

    name: str
    type: str = "string"
    column: str = ""
    description: str | None = None
    expr: str | None = None
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def set_default_column(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("column") and data.get("name"):
            data["column"] = data["name"]
        return data


class MetricModel(BaseModel):
    """Reusable analytical metric with explicit SQL aggregation or derivation formula."""

    name: str
    entity: str | None = None
    type: str = "sum"  # sum, avg, count, count_distinct, min, max, derived, ratio
    sql: str = ""
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)


class RelationshipModel(BaseModel):
    """Relationship between entities."""

    from_entity: str
    to_entity: str
    from_column: str
    to_column: str
    type: str = "many_to_one"  # many_to_one, one_to_many, one_to_one, many_to_many
    description: str | None = None


class EntityModel(BaseModel):
    """Business entity mapped to a physical or logical lakehouse table."""

    name: str
    table_name: str | None = None
    catalog: str | None = None
    schema_name: str | None = None
    primary_key: str | None = None
    foreign_keys: list[Any] | None = Field(default_factory=list)
    dimensions: list[DimensionModel] = Field(default_factory=list)
    metrics: list[MetricModel] = Field(default_factory=list)
    relationships: list[RelationshipModel] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    description: str | None = None
    columns: list[ColumnModel] = Field(default_factory=list)
    layer: str | None = None  # bronze, silver, gold

    @model_validator(mode="before")
    @classmethod
    def normalize_entity(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Support 'table' alias for 'table_name'
        if "table" in data and "table_name" not in data:
            data["table_name"] = data["table"]
        if not data.get("table_name") and data.get("name"):
            data["table_name"] = data["name"]

        # If columns not provided, populate from dimensions and primary_key
        raw_columns = data.get("columns", [])
        if not raw_columns:
            cols: list[dict[str, Any]] = []
            seen = set()
            pk = data.get("primary_key")
            if pk:
                cols.append(
                    {"name": pk, "type": "string", "primary_key": True, "nullable": False}
                )
                seen.add(pk)
            for dim in data.get("dimensions", []):
                dim_name = dim.get("name") if isinstance(dim, dict) else getattr(dim, "name", "")
                dim_col = (
                    dim.get("column", dim_name)
                    if isinstance(dim, dict)
                    else getattr(dim, "column", dim_name)
                )
                dim_type = (
                    dim.get("type", "string")
                    if isinstance(dim, dict)
                    else getattr(dim, "type", "string")
                )
                col_key = dim_col or dim_name
                if col_key and col_key not in seen:
                    cols.append(
                        {
                            "name": col_key,
                            "type": dim_type,
                            "primary_key": (col_key == pk),
                            "nullable": (col_key != pk),
                        }
                    )
                    seen.add(col_key)
            data["columns"] = cols

        return data


class SemanticDomainModel(BaseModel):
    """Domain model grouping entities, metrics, and relationships."""

    domain: str
    description: str | None = None
    catalog: str | None = None
    schema_name: str | None = None
    entities: list[EntityModel] = Field(default_factory=list)
    metrics: list[MetricModel] = Field(default_factory=list)
    relationships: list[RelationshipModel] = Field(default_factory=list)
    synonyms: dict[str, str] = Field(default_factory=dict)
