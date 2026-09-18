# Changelog

Todas as alterações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [0.1.0] - 2026-09-18

### 🚀 Adicionado (Initial Release)

#### 1. Interface Open WebUI & Orquestração LangGraph (R1)
- `open_webui_pipe.py`: Extensão nativa no padrão Open WebUI Pipe com *Valves* dinâmicas para configuração em tempo de execução de credenciais Databricks, URLs de modelos locais e tokens GitHub.
- `src/agent/state.py` e `src/agent/graph.py`: Grafo de estado compilado em LangGraph com nós especializados (`steward_node`), roteamento dinâmico de ferramentas e streaming de tokens.
- `src/agent/tools.py`: Registro de ferramentas do agente (catálogo, semântica, diagramas, ETL, esteira de CI e GitOps).
- Suporte nativo a modelos locais servidos via Ollama ou vLLM (protocolo OpenAI-compatível).
- Servidor REST autônomo em `src/server.py` expondo endpoints `/health`, `/v1/models` e `/v1/chat/completions`.

#### 2. Databricks Unity Catalog & Camada Semântica Declarativa (R2)
- `src/databricks/client.py`: Wrapper sobre o `databricks-sdk` com cliente seguro para catálogo, volumes e execução no SQL Warehouse.
- `src/databricks/introspector.py`: Mapeamento automático de catálogos, schemas, tabelas, colunas, chaves primárias e relacionamentos com modo de demonstração offline resiliente.
- `src/semantic/models.py`: Modelos Pydantic v2 com validação estrita para entidades, dimensões, métricas e relações.
- `src/semantic/registry.py`: Leitor e registro multi-arquivo de ontologias em YAML com resolução automática de sinônimos de negócio.
- `src/semantic/compiler.py`: Resolução de JOINs multi-tabela via busca em grafo (BFS) e compilação de consultas SparkSQL com proteção automática contra divisão por zero (`NULLIF(..., 0)`).
- Modelos semânticos de exemplo prontos para produção: `corporate_credit.yaml` e `sales_lakehouse.yaml`.

#### 3. Visualização e Desenho de Modelagens com Mermaid.js (R3)
- `src/visualizer/mermaid.py`: Gerador de diagramas de Entidade-Relacionamento (`erDiagram`) com notação Crow's foot (`||--o{`, `||--||`) e fluxo medalhão (`graph LR`) com separação visual de subgrafos (Bronze, Silver, Gold).
- Sanitização rigorosa de nomes de tabelas e tipos compatível com os parsers Markdown do Open WebUI.

#### 4. Geração de Pipelines ETL Medalhão (R4)
- `src/etl/generator.py`: Geração modular de scripts PySpark e SparkSQL para as três camadas do Lakehouse:
  - **Bronze**: Ingestão raw, metadados de auditoria e schema enforcement.
  - **Silver**: Limpeza, deduplicação (`row_number()`), enriquecimento e padronização.
  - **Gold**: Agregações analíticas, fatos/dimensões e KPIs de negócio.
- `src/etl/templates.py`: Templates parametrizados para manutenção Delta Lake (`OPTIMIZE`, `ZORDER BY`, `VACUUM` e `MERGE INTO`).

#### 5. Esteira de CI de Boas Práticas de Engenharia de Dados (R5)
- `src/ci/runner.py`: Orquestrador programático da esteira executando validação em memória.
- Integração com **Ruff** para linting de código Python/PySpark.
- Integração com **SQLFluff** configurado com dialeto nativo `sparksql`.
- `src/ci/anti_patterns.py`: Motor de análise estática de anti-patterns de dados:
  - Detecção de `.collect()` sem limitação (`.limit()` ou `.take()`).
  - Detecção de conversão descontrolada para Pandas (`.toPandas()`).
  - Detecção de `CROSS JOIN` e queries SQL `DELETE`/`UPDATE` sem cláusula `WHERE`.
- `src/ci/report.py`: Geração de relatório de auditoria em Markdown estruturado.

#### 6. GitOps Automatizado com GitHub Pull Request (R6)
- `src/gitops/git_client.py`: Criação automatizada de feature branch (`feature/data-product-<nome>`), staging de arquivos gerados e commit com Conventional Commits (`feat(etl): ...`).
- `src/gitops/github_pr.py`: Abertura de Pull Request via PyGithub com template estruturado contendo resumo do produto de dados, diagrama Mermaid e relatório da esteira de CI.
- Trava estrita de segurança: bloqueio automático de abertura de PR se a esteira de CI for reprovada.

#### 7. Infraestrutura, Portabilidade e Containers
- `Dockerfile`: Imagem enxuta baseada em `python:3.11-slim` com usuário não-root (`steward`), saúde (`HEALTHCHECK`), e compatibilidade total com **Docker** e **Podman (rootless)**.
- `docker-compose.yml`: Orquestração unificada contendo `databricks-steward`, `open-webui` e `ollama` (opcional com `--profile with-ollama`).
- `Makefile`: Script com detecção automática de container engine (`docker` vs `podman`) para build, execução, testes e linting.

#### 8. Suíte de Testes Automatizados
- 131 testes automatizados cobrindo testes unitários, testes adversariais (`test_adversarial_fuzzing.py`), regressão e testes ponta a ponta (`test_e2e_scenarios.py`).
