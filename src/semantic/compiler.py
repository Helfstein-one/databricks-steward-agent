"""Semantic query compiler with multi-table graph join resolution and SparkSQL generation."""

from __future__ import annotations

import re
from collections import deque

from src.semantic.models import RelationshipModel
from src.semantic.registry import SemanticRegistry


def ensure_nullif_division_safety(sql_expr: str) -> str:
    """Ensure all division denominators in SQL expression are wrapped with NULLIF(..., 0).

    Handles nested function calls and balanced parentheses such as SUM(column_name)
    without truncating closing parentheses.
    """
    if "/" not in sql_expr:
        return sql_expr

    result: list[str] = []
    i = 0
    n = len(sql_expr)

    while i < n:
        # Preserve string literals
        if sql_expr[i] == "'":
            end_quote = sql_expr.find("'", i + 1)
            if end_quote == -1:
                result.append(sql_expr[i:])
                break
            result.append(sql_expr[i : end_quote + 1])
            i = end_quote + 1
            continue

        # Preserve single-line comments
        if sql_expr[i : i + 2] == "--":
            end_line = sql_expr.find("\n", i + 2)
            if end_line == -1:
                result.append(sql_expr[i:])
                break
            result.append(sql_expr[i : end_line + 1])
            i = end_line + 1
            continue

        # Preserve block comments
        if sql_expr[i : i + 2] == "/*":
            end_comment = sql_expr.find("*/", i + 2)
            if end_comment == -1:
                result.append(sql_expr[i:])
                break
            result.append(sql_expr[i : end_comment + 2])
            i = end_comment + 2
            continue

        # Check for division operator
        if sql_expr[i] == "/":
            result.append("/")
            i += 1
            # Preserve whitespace after slash
            ws_start = i
            while i < n and sql_expr[i].isspace():
                i += 1
            result.append(sql_expr[ws_start:i])

            # Check if already safely wrapped in NULLIF
            rem = sql_expr[i:]
            nullif_match = re.match(r"^NULLIF\s*\(", rem, re.IGNORECASE)
            if nullif_match:
                continue

            # Parse denominator handling balanced parentheses
            denom_chars: list[str] = []
            depth = 0
            stop_keywords = {
                "as",
                "from",
                "where",
                "group",
                "having",
                "order",
                "limit",
                "when",
                "then",
                "else",
                "end",
                "and",
                "or",
                "join",
                "on",
            }

            while i < n:
                ch = sql_expr[i]

                # String literal inside denominator
                if ch == "'":
                    end_q = sql_expr.find("'", i + 1)
                    if end_q == -1:
                        denom_chars.append(sql_expr[i:])
                        i = n
                        break
                    denom_chars.append(sql_expr[i : end_q + 1])
                    i = end_q + 1
                    continue

                if ch == "(":
                    depth += 1
                    denom_chars.append(ch)
                    i += 1
                    continue

                if ch == ")":
                    if depth > 0:
                        depth -= 1
                        denom_chars.append(ch)
                        i += 1
                        continue
                    # Outer closing parenthesis terminates denominator
                    break

                if depth == 0:
                    # Delimiters and operators terminate denominator at depth 0
                    if ch in (",", ";", "\n"):
                        break

                    if ch in ("+", "-", "*", "/"):
                        break

                    if ch in ("=", "<", ">", "!"):
                        break

                    if ch.isspace():
                        peek_match = re.match(r"^\s+([a-zA-Z_]+)\b", sql_expr[i:], re.IGNORECASE)
                        if peek_match:
                            word = peek_match.group(1).lower()
                            if word in stop_keywords:
                                break

                denom_chars.append(ch)
                i += 1

            raw_denom = "".join(denom_chars)
            denom = raw_denom.strip()
            trailing_ws = raw_denom[len(raw_denom.rstrip()) :]
            if denom:
                safe_denom = ensure_nullif_division_safety(denom) if "/" in denom else denom
                result.append(f"NULLIF({safe_denom}, 0){trailing_ws}")
            continue

        result.append(sql_expr[i])
        i += 1

    return "".join(result)


class GraphJoinResolver:
    """Resolves shortest join paths between entities in the semantic relationship graph."""

    def __init__(self, relationships: list[RelationshipModel]):
        self.relationships = relationships
        self.graph: dict[str, list[tuple[str, RelationshipModel, bool]]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build bidirectional adjacency list representing relationships."""
        for rel in self.relationships:
            from_ent = rel.from_entity.lower()
            to_ent = rel.to_entity.lower()

            if from_ent not in self.graph:
                self.graph[from_ent] = []
            if to_ent not in self.graph:
                self.graph[to_ent] = []

            # forward: True means from_entity -> to_entity
            self.graph[from_ent].append((to_ent, rel, True))
            # backward: False means to_entity -> from_entity
            self.graph[to_ent].append((from_ent, rel, False))

    def find_shortest_path(
        self, start_entity: str, target_entity: str
    ) -> list[tuple[str, RelationshipModel, bool]] | None:
        """Find the shortest join path between start and target entity using BFS."""
        start = start_entity.lower()
        target = target_entity.lower()

        if start == target:
            return []

        queue: deque[tuple[str, list[tuple[str, RelationshipModel, bool]]]] = deque([(start, [])])
        visited = {start}

        while queue:
            curr_entity, path = queue.popleft()

            for neighbor, rel, is_forward in self.graph.get(curr_entity, []):
                if neighbor == target:
                    return path + [(neighbor, rel, is_forward)]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, rel, is_forward)]))

        return None


class SemanticQueryCompiler:
    """Compiles analytical business requests into optimized SparkSQL queries."""

    def __init__(self, registry: SemanticRegistry):
        self.registry = registry
        self.join_resolver = GraphJoinResolver(registry.list_relationships())

    def compile_query(
        self,
        entity_name: str,
        metric_names: list[str] | None = None,
        group_by_dims: list[str] | None = None,
        filters: list[str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> str:
        """Compile a list of metrics and dimensions into a SparkSQL query.

        Args:
            entity_name: Primary base entity name.
            metric_names: List of analytical metric names to compute.
            group_by_dims: List of dimensions to project and group by.
            filters: WHERE clause filter conditions.
            order_by: ORDER BY expressions.
            limit: Query result row limit.

        Returns:
            Formatted SparkSQL string.
        """
        metric_names = metric_names or []
        group_by_dims = group_by_dims or []
        filters = filters or []
        order_by = order_by or []

        if not metric_names and not group_by_dims:
            raise ValueError("At least one metric or dimension must be requested.")

        primary_entity = self.registry.get_entity(entity_name)
        if not primary_entity:
            raise ValueError(f"Entity '{entity_name}' not found in semantic registry.")

        primary_table = primary_entity.table_name or primary_entity.name
        primary_alias = primary_entity.name.lower()

        # Track entities involved in the query to resolve joins
        entities_involved: set[str] = {primary_alias}
        projections: list[str] = []
        group_by_cols: list[str] = []

        # Process dimensions
        for dim_name in group_by_dims:
            dim_info = self.registry.find_dimension_owner(
                dim_name, preferred_entity=primary_entity.name
            )
            if dim_info:
                dim, owner_entity = dim_info
                owner_alias = owner_entity.lower()
                entities_involved.add(owner_alias)
                col_name = dim.column or dim.name
                col_ref = f"{owner_alias}.{col_name}"
                projections.append(f"{col_ref} AS {dim.name}")
                group_by_cols.append(col_ref)
            else:
                # Treat as raw column on primary table
                col_ref = f"{primary_alias}.{dim_name}"
                projections.append(f"{col_ref} AS {dim_name}")
                group_by_cols.append(col_ref)

        # Process metrics
        for m_name in metric_names:
            metric_info = self.registry.find_metric_owner(
                m_name, preferred_entity=primary_entity.name
            )
            if metric_info:
                metric, owner_entity = metric_info
                if owner_entity:
                    entities_involved.add(owner_entity.lower())
                sql_expr = ensure_nullif_division_safety(metric.sql)
                projections.append(f"{sql_expr} AS {metric.name}")
            else:
                # Unknown metric: pass through
                projections.append(f"{m_name} AS {m_name}")

        # Resolve Joins
        joins: list[str] = []
        joined_entities = {primary_alias}

        for target_ent in entities_involved:
            if target_ent not in joined_entities:
                path = self.join_resolver.find_shortest_path(primary_alias, target_ent)
                if path is None:
                    raise ValueError(
                        f"No join path found in semantic model between '{primary_alias}' and requested entity '{target_ent}'."
                    )
                for neighbor, rel, is_forward in path:
                    if neighbor not in joined_entities:
                        target_entity_model = self.registry.get_entity(neighbor)
                        target_table = (
                            target_entity_model.table_name
                            if target_entity_model and target_entity_model.table_name
                            else neighbor
                        )
                        if is_forward:
                            # from_entity -> to_entity (neighbor)
                            from_alias = rel.from_entity.lower()
                            joins.append(
                                f"LEFT JOIN {target_table} AS {neighbor} "
                                f"ON {from_alias}.{rel.from_column} = {neighbor}.{rel.to_column}"
                            )
                        else:
                            # to_entity -> from_entity (neighbor)
                            to_alias = rel.to_entity.lower()
                            joins.append(
                                f"LEFT JOIN {target_table} AS {neighbor} "
                                f"ON {to_alias}.{rel.to_column} = {neighbor}.{rel.from_column}"
                            )
                        joined_entities.add(neighbor)

        # Build SQL statements
        select_clause = "SELECT\n  " + ",\n  ".join(projections)
        from_clause = f"FROM {primary_table} AS {primary_alias}"

        sql_parts = [select_clause, from_clause]
        if joins:
            sql_parts.extend(joins)

        if filters:
            sql_parts.append("WHERE " + " AND ".join(filters))

        if group_by_cols and metric_names:
            sql_parts.append("GROUP BY " + ", ".join(group_by_cols))

        if order_by:
            sql_parts.append("ORDER BY " + ", ".join(order_by))

        if limit is not None:
            sql_parts.append(f"LIMIT {limit}")

        return "\n".join(sql_parts)
