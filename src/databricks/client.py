"""Databricks SDK workspace and SQL warehouse client wrapper."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

from src.config import settings

logger = logging.getLogger(__name__)


class DatabricksClientError(Exception):
    """Base exception for Databricks client errors."""


class DatabricksConnectionError(DatabricksClientError):
    """Raised when connection to Databricks workspace fails."""


class DatabricksCEClient:
    """Wrapper around Databricks WorkspaceClient providing workspace and SQL execution."""

    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        warehouse_id: str | None = None,
        timeout: int = 60,
    ):
        self.host = (host or settings.databricks_host).rstrip("/")
        self.token = token or settings.databricks_token
        self.warehouse_id = warehouse_id or settings.databricks_warehouse_id
        self.timeout = timeout
        self.client: WorkspaceClient | None = None
        self._init_error: str | None = None

        if self.host and self.token:
            try:
                self.client = WorkspaceClient(
                    host=self.host,
                    token=self.token,
                )
            except Exception as e:  # noqa: BLE001
                self.client = None
                self._init_error = str(e)
        else:
            self._init_error = "Databricks host or token is not configured."

    def is_configured(self) -> bool:
        """Return True if client has credentials configured."""
        return self.client is not None and bool(self.host and self.token)

    def verify_connection(self) -> dict[str, Any]:
        """Verify connection to Databricks workspace."""
        if not self.client:
            raise DatabricksConnectionError(f"Databricks client not initialized: {self._init_error}")

        try:
            user = self.client.current_user.me()
            return {
                "status": "connected",
                "user": getattr(user, "user_name", "unknown"),
                "host": self.host,
            }
        except Exception as e:
            raise DatabricksConnectionError(f"Failed to connect to Databricks: {e}") from e

    def mkdirs(self, remote_workspace_path: str) -> dict[str, Any]:
        """Create remote directory in Databricks workspace."""
        if not self.client:
            raise DatabricksConnectionError("Databricks client not initialized.")
        try:
            self.client.workspace.mkdirs(path=remote_workspace_path)
            return {"status": "success", "path": remote_workspace_path}
        except Exception as e:
            raise DatabricksClientError(f"Failed to create directory {remote_workspace_path}: {e}") from e

    def upload_file(
        self,
        local_path: str | Path,
        remote_workspace_path: str,
        file_format: str | None = None,
        language: str = "PYTHON",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Upload a local file to Databricks workspace."""
        if not self.client:
            raise DatabricksConnectionError("Databricks client not initialized.")

        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_file}")

        # Ensure parent directory exists
        parent_dir = os.path.dirname(remote_workspace_path)
        if parent_dir and parent_dir != "/":
            try:
                self.mkdirs(parent_dir)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Parent directory creation notice: {exc}")

        fmt = ImportFormat.SOURCE
        if file_format:
            fmt_upper = file_format.upper()
            if fmt_upper in ImportFormat.__members__:
                fmt = ImportFormat[fmt_upper]

        try:
            with open(local_file, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

            self.client.workspace.import_(
                path=remote_workspace_path,
                format=fmt,
                language=language,
                content=content,
                overwrite=overwrite,
            )
            return {"status": "success", "path": remote_workspace_path}
        except Exception as e:
            raise DatabricksClientError(f"Failed to upload file to {remote_workspace_path}: {e}") from e

    def list_status(self, remote_workspace_path: str) -> list[dict[str, Any]]:
        """List objects in Databricks workspace path."""
        if not self.client:
            raise DatabricksConnectionError("Databricks client not initialized.")
        try:
            items = list(self.client.workspace.list(path=remote_workspace_path))
            return [
                {
                    "path": getattr(item, "path", None),
                    "object_type": str(getattr(item, "object_type", None)),
                }
                for item in items
            ]
        except Exception as e:
            raise DatabricksClientError(f"Failed to list path {remote_workspace_path}: {e}") from e

    def delete_path(self, remote_workspace_path: str, recursive: bool = False) -> dict[str, Any]:
        """Delete path in Databricks workspace."""
        if not self.client:
            raise DatabricksConnectionError("Databricks client not initialized.")
        try:
            self.client.workspace.delete(path=remote_workspace_path, recursive=recursive)
            return {"status": "deleted", "path": remote_workspace_path}
        except Exception as e:
            raise DatabricksClientError(f"Failed to delete path {remote_workspace_path}: {e}") from e

    def execute_query(self, sql_query: str, warehouse_id: str | None = None) -> dict[str, Any]:
        """Execute a SQL statement on a Databricks SQL warehouse."""
        if not self.client:
            raise DatabricksConnectionError("Databricks client not initialized.")

        target_wh = warehouse_id or self.warehouse_id
        if not target_wh:
            raise DatabricksClientError("No Databricks warehouse_id configured or provided.")

        try:
            response = self.client.statement_execution.execute_statement(
                statement=sql_query,
                warehouse_id=target_wh,
                wait_timeout="30s",
            )
            return {
                "statement_id": getattr(response, "statement_id", None),
                "status": getattr(getattr(response, "status", None), "state", "UNKNOWN"),
                "result": getattr(response, "result", None),
            }
        except Exception as e:
            raise DatabricksClientError(f"Failed to execute SQL query: {e}") from e
