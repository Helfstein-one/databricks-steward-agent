.PHONY: help install test lint format run build up up-ollama down logs status clean docker-build podman-build

# Container Engine Detection (Docker vs Podman)
DOCKER_BIN := $(shell which docker 2>/dev/null)
PODMAN_BIN := $(shell which podman 2>/dev/null)

ifneq ($(PODMAN_BIN),)
    CONTAINER_ENGINE ?= podman
    COMPOSE ?= podman compose
else ifneq ($(DOCKER_BIN),)
    CONTAINER_ENGINE ?= docker
    COMPOSE ?= docker compose
else
    CONTAINER_ENGINE ?= docker
    COMPOSE ?= docker compose
endif

IMAGE_NAME ?= databricks-steward-agent:latest
PYTHON ?= $(shell test -f .venv/bin/python && echo .venv/bin/python || which python3 || echo python)

help:
	@echo "=========================================================================="
	@echo " Databricks Steward Agent - Automação e Gerenciamento"
	@echo " Container Engine detectado: $(CONTAINER_ENGINE) (Compose: $(COMPOSE))"
	@echo " Python detectado:           $(PYTHON)"
	@echo "=========================================================================="
	@echo "Comandos Locais:"
	@echo "  make install     - Instala dependências Python locais"
	@echo "  make test        - Executa toda a suíte de testes com pytest"
	@echo "  make lint        - Executa validação de código com Ruff"
	@echo "  make format      - Formata o código com Ruff"
	@echo "  make run         - Inicia o servidor FastAPI local na porta 8000"
	@echo ""
	@echo "Comandos Docker / Podman:"
	@echo "  make build       - Constrói a imagem ($(CONTAINER_ENGINE) build)"
	@echo "  make up          - Inicia a stack completa (Agent + Open WebUI)"
	@echo "  make up-ollama   - Inicia a stack com Ollama containerizado (--profile with-ollama)"
	@echo "  make down        - Para todos os containers da stack"
	@echo "  make logs        - Exibe os logs dos containers em tempo real"
	@echo "  make status      - Verifica o status dos containers ativos"
	@echo "  make docker-build- Força build usando Docker especificamente"
	@echo "  make podman-build- Força build usando Podman especificamente (suporte a rootless)"
	@echo ""
	@echo "Manutenção:"
	@echo "  make clean       - Remove caches (__pycache__, .pytest_cache, .ruff_cache)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -v tests/

lint:
	$(PYTHON) -m ruff check src/

format:
	$(PYTHON) -m ruff format src/

run:
	$(PYTHON) -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

build:
	$(CONTAINER_ENGINE) build -t $(IMAGE_NAME) .

docker-build:
	docker build -t $(IMAGE_NAME) .

podman-build:
	podman build -t $(IMAGE_NAME) .

up:
	$(COMPOSE) up -d

up-ollama:
	$(COMPOSE) --profile with-ollama up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

status:
	$(COMPOSE) ps

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info .coverage
