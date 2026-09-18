"""Unit and component integration tests for GitOps automation and PyGithub PR (Tier 2)."""

from unittest.mock import MagicMock

import pytest

from src.ci.report import CIReport
from src.gitops.git_client import GitClient
from src.gitops.github_pr import GitHubPRManager, GitOpsResult, create_data_product_pr


@pytest.fixture
def mock_git_client(tmp_path):
    """Provides a GitClient operating in an isolated temporary directory with mocked git commands."""
    client = GitClient(repo_path=tmp_path)
    client._run_git = MagicMock()
    # Mock checkout and commit success
    proc_ok = MagicMock(returncode=0, stdout="a1b2c3d4e5f6\n", stderr="")
    client._run_git.return_value = proc_ok
    return client


@pytest.fixture
def approved_ci_report():
    return CIReport(
        is_approved=True,
        ruff_status="PASSED",
        sqlfluff_status="PASSED",
        anti_patterns=[],
        violations=[],
        summary_markdown="### ✅ CI Quality Gate: 100% Approved",
    )


@pytest.fixture
def rejected_ci_report():
    return CIReport(
        is_approved=False,
        ruff_status="FAILED",
        sqlfluff_status="PASSED",
        anti_patterns=[],
        violations=[],
        summary_markdown="### ❌ CI Quality Gate: Rejected",
    )


# ==============================================================================
# GitClient Tests
# ==============================================================================


def test_git_client_create_feature_branch(mock_git_client):
    """Verify branch naming follows feature/data-product-<name> format."""
    branch = mock_git_client.create_feature_branch("Credit Risk Lakehouse")
    assert branch == "feature/data-product-credit-risk-lakehouse"
    mock_git_client._run_git.assert_called()


def test_git_client_commit_files(mock_git_client, tmp_path):
    """Verify writing files and generating Conventional Commits."""
    files = {
        "src/pipeline.py": "print('hello')",
        "models/credit.sql": "SELECT 1",
    }
    sha = mock_git_client.commit_files(
        branch_name="feature/data-product-test",
        files=files,
    )
    assert sha == "a1b2c3d4e5f6"

    # Verify files were actually written to tmp_path
    assert (tmp_path / "src/pipeline.py").exists()
    assert (tmp_path / "models/credit.sql").exists()
    assert (tmp_path / "src/pipeline.py").read_text() == "print('hello')"


# ==============================================================================
# GitHub PR Manager Tests
# ==============================================================================


def test_build_pr_body(approved_ci_report):
    """Verify PR description embeds artifacts, Mermaid diagram, and CI report."""
    manager = GitHubPRManager()
    files = {"pipeline.py": "...", "model.sql": "..."}
    diagram = "erDiagram\n    tbl_a ||--o{ tbl_b : has"

    body = manager.build_pr_body(
        product_name="credit-risk",
        files=files,
        ci_report=approved_ci_report,
        diagram_md=diagram,
    )

    assert "# Data Product: Credit Risk" in body
    assert "- `pipeline.py`" in body
    assert "- `model.sql`" in body
    assert "```mermaid" in body
    assert "erDiagram" in body
    assert "100% Approved" in body


def test_gitops_strictly_blocks_on_rejected_ci(rejected_ci_report):
    """Verify GitOps strictly blocks branch and PR creation if CI failed."""
    result = create_data_product_pr(
        product_name="bad-product",
        files={"bad.py": "bad_code"},
        ci_report=rejected_ci_report,
    )

    assert isinstance(result, GitOpsResult)
    assert result.status == "rejected"
    assert result.pr_number is None
    assert "rejected" in result.body.lower()


def test_gitops_dry_run_success(approved_ci_report):
    """Verify GitOps dry-run generates branch, commit, and simulated PR metadata."""
    result = create_data_product_pr(
        product_name="wholesale-credit",
        files={"pipeline.py": "clean_code"},
        ci_report=approved_ci_report,
        dry_run=True,
    )

    assert result.status == "success"
    assert result.branch_name == "feature/data-product-wholesale-credit"
    assert result.commit_sha != ""
    assert result.pr_number == 101
    assert "pull/101" in result.pr_url


def test_gitops_with_mock_github_repo(approved_ci_report, mock_github, mock_git_client):
    """Verify PyGithub PR creation with mock repo and branch."""
    _mock_gh, mock_repo, _mock_pr = mock_github

    manager = GitHubPRManager(
        token="test-token",
        repository="test-org/lakehouse",
        git_client=mock_git_client,
    )

    result = manager.create_data_product_pr(
        product_name="retail-sales",
        files={"sales.py": "def process(): pass"},
        ci_report=approved_ci_report,
        diagram_md="graph LR\n    Bronze --> Silver",
        repo=mock_repo,
    )

    assert result.status == "success"
    assert result.pr_number == 42
    assert result.pr_url == "https://github.com/test-org/test-data-lakehouse/pull/42"

    mock_repo.create_pull.assert_called_once()
    kwargs = mock_repo.create_pull.call_args[1]
    assert kwargs["head"] == "feature/data-product-retail-sales"
    assert kwargs["base"] == "main"
    assert "retail-sales" in kwargs["title"]
    assert "```mermaid" in kwargs["body"]


def test_gitops_github_api_failure_handling(approved_ci_report, mock_git_client):
    """Verify graceful handling and status='failed' when PyGithub raises exception."""
    mock_repo = MagicMock()
    mock_repo.create_pull.side_effect = RuntimeError("GitHub 403 Forbidden")

    manager = GitHubPRManager(
        token="test-token",
        repository="test-org/lakehouse",
        git_client=mock_git_client,
    )

    result = manager.create_data_product_pr(
        product_name="failing-pr",
        files={"test.py": "pass"},
        ci_report=approved_ci_report,
        repo=mock_repo,
    )

    assert result.status == "failed"
    assert "403 Forbidden" in result.body
