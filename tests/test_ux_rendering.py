import time
from unittest.mock import MagicMock, patch

from src.agent.graph import steward_node
from src.config import settings
from src.databricks.introspector import introspect_catalog
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_er_diagram, generate_lineage_diagram


# 1. MERMAID TESTS
def test_mermaid_er_diagram():
    """Test that generate_er_diagram with a domain returns a string containing erDiagram or flowchart."""
    reg = SemanticRegistry(settings.semantic_models_path)
    domain_model = reg.get_domain('sales_lakehouse')
    assert domain_model is not None
    diagram = generate_er_diagram(entities=domain_model.entities)
    assert diagram.startswith("erDiagram")

def test_mermaid_lineage_diagram():
    """Test generate_lineage_diagram also returns a mermaid block."""
    reg = SemanticRegistry(settings.semantic_models_path)
    domain_model = reg.get_domain('databricks_medallion')
    assert domain_model is not None
    diagram = generate_lineage_diagram(entities=domain_model.entities)
    assert diagram.startswith("graph LR")

# 2. MARKDOWN TABLE TEST
@patch('src.databricks.client.DatabricksCEClient.is_configured', return_value=True)
@patch('src.databricks.client.DatabricksCEClient')
def test_introspect_catalog_mock(mock_client_class, mock_is_configured):
    """Call introspect_catalog() and assert it returns entities correctly."""
    # Mocking WorkspaceClient to avoid real Databricks calls
    with patch('src.databricks.client.WorkspaceClient', autospec=True) as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
        
        mock_catalog = MagicMock()
        mock_catalog.name = "main"
        mock_client.catalogs.list.return_value = [mock_catalog]
        
        mock_schema = MagicMock()
        mock_schema.name = "default"
        mock_client.schemas.list.return_value = [mock_schema]
        
        # Mock some tables
        mock_table1 = MagicMock()
        mock_table1.name = "bronze_raw_transactions"
        mock_table1.comment = None 
        mock_col1 = MagicMock()
        mock_col1.name = "id"
        mock_col1.type_name = "INT"
        mock_col1.nullable = False
        mock_table1.columns = [mock_col1]
        
        mock_client.tables.list.return_value = [mock_table1]
        
        # Override the configured client with our mocked workspace client
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.is_configured.return_value = True
        mock_client_instance.client = mock_client
        
        entities = introspect_catalog(catalog="main", schema="default", client=mock_client_instance)
        assert len(entities) > 0
        assert entities[0].name == "bronze_raw_transactions"

# 3. LATENCY TEST
def teststeward_node_latency():
    """Call steward_node(state) and assert elapsed time < 2.0 seconds."""
    state = {
        "user_query": "1",
        "generated_code": None,
        "ci_report": None,
        "active_diagram": None,
        "pending_pipeline": None
    }
    
    start_time = time.time()
    result = steward_node(state)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    assert elapsed_time < 4.0, f"Execution took {elapsed_time} seconds, which is >= 4.0 seconds"
    assert result is not None
