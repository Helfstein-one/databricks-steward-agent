"""Delta Lake optimization and maintenance SQL templates."""

from __future__ import annotations


def build_optimize_query(
    table_name: str,
    zorder_columns: list[str] | None = None,
    where_clause: str | None = None,
) -> str:
    """Build Delta Lake OPTIMIZE statement with optional ZORDER BY and partition filter."""
    parts = [f"OPTIMIZE {table_name}"]

    if where_clause:
        parts.append(f"WHERE {where_clause}")

    if zorder_columns:
        zorder_str = ", ".join(zorder_columns)
        parts.append(f"ZORDER BY ({zorder_str})")

    return " ".join(parts) + ";"


def build_vacuum_query(table_name: str, retention_hours: int = 168) -> str:
    """Build Delta Lake VACUUM statement with retention period."""
    return f"VACUUM {table_name} RETAIN {retention_hours} HOURS;"


def build_delta_create_table(
    table_name: str,
    columns: list[dict[str, str]],
    partition_by: list[str] | None = None,
    comment: str | None = None,
) -> str:
    """Build idempotent CREATE TABLE IF NOT EXISTS statement for Delta Lake."""
    col_defs = [f"  `{col['name']}` {col.get('type', 'STRING').upper()}" for col in columns]
    cols_block = ",\n".join(col_defs)

    query = f"CREATE TABLE IF NOT EXISTS {table_name} (\n{cols_block}\n) USING DELTA"

    if partition_by:
        parts_str = ", ".join([f"`{p}`" for p in partition_by])
        query += f"\nPARTITIONED BY ({parts_str})"

    if comment:
        clean_comment = comment.replace("'", "''")
        query += f"\nCOMMENT '{clean_comment}'"

    return query + ";"


def build_delta_merge_query(
    target_table: str,
    source_table: str,
    join_keys: list[str],
    update_columns: list[str] | None = None,
    insert_columns: list[str] | None = None,
) -> str:
    """Build idempotent Delta MERGE INTO statement."""
    join_conditions = " AND ".join([f"target.`{k}` = source.`{k}`" for k in join_keys])

    if update_columns:
        set_statements = ", ".join([f"`{c}` = source.`{c}`" for c in update_columns])
        matched_clause = f"WHEN MATCHED THEN UPDATE SET {set_statements}"
    else:
        matched_clause = "WHEN MATCHED THEN UPDATE SET *"

    if insert_columns:
        insert_cols = ", ".join([f"`{c}`" for c in insert_columns])
        insert_vals = ", ".join([f"source.`{c}`" for c in insert_columns])
        not_matched_clause = f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    else:
        not_matched_clause = "WHEN NOT MATCHED THEN INSERT *"

    return f"""MERGE INTO {target_table} AS target
USING {source_table} AS source
ON {join_conditions}
{matched_clause}
{not_matched_clause};"""
