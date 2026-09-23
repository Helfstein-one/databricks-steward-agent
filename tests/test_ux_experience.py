import time

import pytest

from open_webui_pipe import Pipe


@pytest.fixture(scope="module")
def sample_ux_response():
    """Fixture to execute the Open WebUI pipe once and share the response across UX tests."""
    pipe = Pipe()
    t0 = time.time()
    pipe_response = pipe.pipe(
        {
            "messages": [{"role": "user", "content": "Desenhar modelo de diagramas e arquitetura"}],
            "stream": False,
        }
    )
    elapsed = time.time() - t0
    return {"response": pipe_response, "elapsed": elapsed}


def test_markdown_rendering(sample_ux_response):
    """Test Markdown rendering experience."""
    pipe_response = sample_ux_response["response"]
    assert "### 📊 Modelo de Entidade-Relacionamento" in pipe_response
    assert "#### 🔗 Relações e Chaves Estrangeiras" in pipe_response
    assert (
        "| Tabela Origem | Chave Origem (FK) | Relacionamento | Tabela Destino | Chave Destino (PK) |"
        in pipe_response
    )


def test_diagrams_rendering(sample_ux_response):
    """Test Diagrams rendering experience."""
    pipe_response = sample_ux_response["response"]
    assert "```mermaid" in pipe_response
    assert "erDiagram" in pipe_response


def test_e2e_latency_validation(sample_ux_response):
    """Test E2E Latency validation."""
    elapsed = sample_ux_response["elapsed"]
    assert elapsed < 5.0
