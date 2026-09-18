# Original User Request

## 2026-09-18T00:59:05Z

Construir o **Databricks Steward Agent**, uma solução completa de governança e engenharia de dados GenAI com suporte a modelos locais integrada ao Open WebUI. O agente consulta o Databricks (Unity Catalog & SQL Warehouse), expõe uma camada semântica com diagramas Mermaid, desenha produtos de dados, gera pipelines modulares em PySpark/SparkSQL, submete o código a uma esteira de CI de boas práticas de dados e automatiza o ciclo GitOps criando branch, commit, push e Pull Request no GitHub.

Working directory: /Users/mauriciohelfstein/dev/databricks-steward-agent
Integrity mode: demo

## Reference Projects
- `/Users/mauriciohelfstein/dev/databricks-forge-cli`: Modelos de conexão Databricks SDK (`client.py`), CI runner (`ci_runner.py`), GitOps (`git_ops.py`), e utilitários SQL.
- `/Users/mauriciohelfstein/dev/icepol-semantic`: Registro semântico (`semantic/parser.py`), ontologias YAML com dimensões/métricas/joins e compilação de consultas.

## Requirements

### R1. Interface Open WebUI e Modelos Locais
Fornecer um script de pipeline compatível com Open WebUI (`Open WebUI Pipe`) que conecte a modelos locais (Ollama / vLLM via protocolo compatível com OpenAI) com orquestração de ferramentas e LangGraph.

### R2. Databricks Unity Catalog & Camada Semântica
Integrar com Databricks SDK para inspeção de catálogos, schemas, tabelas e chaves, combinando com uma camada semântica declarativa em YAML (definindo entidades, dimensões, métricas analíticas, sinônimos de negócio e relacionamentos).

### R3. Visualização e Desenho de Modelagens com Mermaid.js
Gerar diagramas nativos em formato Mermaid.js (`erDiagram` e diagramas de linhagem/fluxo) diretamente renderizáveis no chat do Open WebUI, permitindo ao usuário visualizar esquemas e desenhar produtos de dados de forma conversacional.

### R4. Geração de Pipelines ETL Modulares
Gerar código de engenharia de dados em PySpark ou SparkSQL estruturado com separação de camadas medalhão (Bronze, Silver, Gold), tipagem explícita, idempotência e modularização.

### R5. Esteira de CI de Boas Práticas de Dados
Executar uma esteira de validação antes de qualquer commit contendo:
- Linting e formatação com Ruff para arquivos Python / PySpark.
- Linting com SQLFluff configurado para o dialeto `sparksql`.
- Análise de boas práticas de dados (detecção de anti-patterns como ausência de particionamento, cross-joins acidentais ou chamadas de `collect()` perigosas).
- Emissão de relatório estruturado com status de aprovação/reprovação.

### R6. GitOps Automatizado com GitHub PR
Após a aprovação na esteira de CI, criar automaticamente uma feature branch (`feature/data-product-<nome>`), realizar commit com mensagem estruturada (Conventional Commits), push remoto e abertura de Pull Request no repositório GitHub configurado, incluindo o relatório de CI na descrição.

## Acceptance Criteria

### Testes Automatizados e Cobertura
- [ ] Execução bem-sucedida de `pytest tests/` com cobertura completa das funcionalidades sem erros.
- [ ] Testes unitários validando a carga e resolução da camada semântica YAML.
- [ ] Testes unitários validando a geração correta de diagramas Mermaid (`erDiagram`).
- [ ] Testes da esteira de CI verificando aprovação de código limpo e rejeição de código com erros sintáticos ou anti-patterns.
- [ ] Testes com mocks para as operações de GitOps (criação de branch, commit e PR via PyGithub).

### Operação e Entrega
- [ ] Script de pipeline pronto para importação no Open WebUI.
- [ ] Arquivo de configuração `.env.example` e documentação de uso no `README.md`.
- [ ] Modelos semânticos de exemplo prontos para uso em `configs/semantic_models/`.
