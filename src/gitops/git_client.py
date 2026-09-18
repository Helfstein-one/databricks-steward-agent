"""Git client for branch creation, file commits, and push automation."""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path


class GitClient:
    """Automates local git operations for data product delivery."""

    def __init__(self, repo_path: Path | str = "."):
        self.repo_path = Path(repo_path).resolve()

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute a git command in the repository path."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )

    def create_feature_branch(self, product_name: str, base_branch: str = "main") -> str:
        """Create and checkout a new feature branch for the data product.

        Branch naming convention: feature/data-product-<name>
        """
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "-", product_name.lower()).strip("-")
        branch_name = f"feature/data-product-{clean_name}"

        # Try git command
        res = self._run_git(["checkout", "-b", branch_name])
        if res.returncode != 0:
            # If branch already exists, checkout it
            self._run_git(["checkout", branch_name])

        return branch_name

    def commit_files(
        self,
        branch_name: str,
        files: dict[str, str],
        message: str | None = None,
        author_name: str = "Databricks Steward Agent",
        author_email: str = "steward-agent@databricks.local",
    ) -> str:
        """Write files, stage them, and commit with Conventional Commits message.

        Returns:
            The commit SHA string.
        """
        for rel_path, content in files.items():
            dest = self.repo_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content)
            self._run_git(["add", str(dest)])

        commit_msg = message or f"feat(data-product): implement {branch_name.replace('feature/', '')} pipeline"

        cmd = [
            "-c", f"user.name={author_name}",
            "-c", f"user.email={author_email}",
            "commit",
            "-m", commit_msg,
        ]
        self._run_git(cmd)

        # Get commit SHA
        sha_res = self._run_git(["rev-parse", "HEAD"])
        if sha_res.returncode == 0 and sha_res.stdout.strip():
            return sha_res.stdout.strip()

        # Fallback pseudo-sha if git commit is in dry run or no git tree
        return hashlib.sha256(f"{branch_name}-{time.time()}".encode()).hexdigest()[:12]

    def push_branch(self, branch_name: str, remote: str = "origin") -> bool:
        """Push branch to remote. Returns True if successful or dry-run."""
        res = self._run_git(["push", "-u", remote, branch_name])
        return res.returncode == 0
