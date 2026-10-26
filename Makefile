#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mi-bici
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mi_bici/dataset.py


CONFIG = $(shell $(PYTHON_INTERPRETER) -c "from mi_bici.config import $(1); print($(1))")
MLFLOW_BACKEND_STORE_URI = $(call CONFIG,MLFLOW_BACKEND_STORE_URI)
MLFLOW_EXPERIMENT_NAME = $(call CONFIG,MLFLOW_EXPERIMENT_NAME)
MLFLOW_SERVER_ADDRESS = $(shell $(PYTHON_INTERPRETER) -c "from urllib.parse import urlparse; \
	from mi_bici.config import MLFLOW_TRACKING_URI; u = urlparse(MLFLOW_TRACKING_URI); \
	print(f'--host {u.hostname} --port {u.port}')")
MODELS = $(shell $(PYTHON_INTERPRETER) -c "from mi_bici.config import MODEL_DEFAULTS as d; \
	print(' '.join(f'{q}:{m}' for q in d for m in d[q]))")

# `mlflow run` toma el servidor de esta variable
ifndef MLFLOW_TRACKING_URI
MLFLOW_TRACKING_URI := $(call CONFIG,MLFLOW_TRACKING_URI)
endif
export MLFLOW_TRACKING_URI

# Entorno de `mlflow run`. uv (aislado, desde python_env.yaml) o local (entorno activo)
ENV_MANAGER ?= uv


## Arrancar el MLflow tracking server
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri $(MLFLOW_BACKEND_STORE_URI) $(MLFLOW_SERVER_ADDRESS)


## Entrenar cada modelo
.PHONY: train-all
train-all:
	@for spec in $(MODELS); do \
		echo ">>> mlflow run -e train -P question=$${spec%%:*} -P model=$${spec#*:}"; \
		mlflow run . -e train -P question=$${spec%%:*} -P model=$${spec#*:} \
			--experiment-name $(MLFLOW_EXPERIMENT_NAME) --env-manager $(ENV_MANAGER) || exit 1; \
	done


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
