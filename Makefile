#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops_equipo19
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mlops
	isort --check --diff mlops
	black --check mlops

## Format source code with black
.PHONY: format
format:
	isort mlops
	black mlops



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mlops/dataset.py


#################################################################################
# MLFLOW COMMANDS                                                               #
#################################################################################

## Train models with MLflow tracking (single run)
.PHONY: train-mlflow
train-mlflow:
	$(PYTHON_INTERPRETER) src/train_with_mlflow.py

## Run multiple experiments with MLflow tracking
.PHONY: train-mlflow-multi
train-mlflow-multi:
	$(PYTHON_INTERPRETER) src/train_multiple_experiments_mlflow.py

## Run grid search experiments with MLflow tracking
.PHONY: train-mlflow-grid
train-mlflow-grid:
	$(PYTHON_INTERPRETER) src/train_grid_search_mlflow.py

## DVC + S3 Commands
setup-dvc-s3:
	@echo "Setting up DVC with S3..."
	$(PYTHON_INTERPRETER) scripts/setup_dvc_s3.py

dvc-push-s3:
	@echo "Pushing DVC data to S3..."
	$(PYTHON_INTERPRETER) -c "from mlops.mlflow import MLflowManager; MLflowManager().push_data_to_s3()"

dvc-pull-s3:
	@echo "Pulling DVC data from S3..."
	$(PYTHON_INTERPRETER) -c "from mlops.mlflow import MLflowManager; MLflowManager().pull_data_from_s3()"

example-dvc-s3:
	@echo "Running DVC + S3 integration example..."
	$(PYTHON_INTERPRETER) examples/dvc_s3_integration_example.py

## Start MLflow UI server
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --port 5000

## List MLflow experiments
.PHONY: mlflow-list
mlflow-list:
	mlflow experiments list


#################################################################################
# API COMMANDS                                                                  #
#################################################################################

## Train models for API
.PHONY: train-api
train-api:
	$(PYTHON_INTERPRETER) src/train_models_for_api.py

## Run API server
.PHONY: run-api
run-api:
	$(PYTHON_INTERPRETER) src/api/run_server.py

## Run API server with uvicorn directly
.PHONY: run-api-uvicorn
run-api-uvicorn:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

## Stop API server
.PHONY: stop-api
stop-api:
	@pkill -f "python.*run_server" || pkill -f "uvicorn.*src.api.main" || echo "No API server process found"


#################################################################################
# REPRODUCIBILITY COMMANDS                                                      #
#################################################################################

## Save reference metrics for reproducibility validation
.PHONY: save-reference-metrics
save-reference-metrics:
	$(PYTHON_INTERPRETER) scripts/validate_reproducibility.py \
		--model-type random_forest \
		--save-reference reference_metrics.json

## Validate reproducibility against reference metrics
.PHONY: validate-reproducibility
validate-reproducibility:
	@if [ ! -f reference_metrics.json ]; then \
		echo "Error: reference_metrics.json not found. Run 'make save-reference-metrics' first."; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) scripts/validate_reproducibility.py \
		--model-type random_forest \
		--reference-metrics-file reference_metrics.json \
		--tolerance 1e-6

## Validate reproducibility using MLflow run ID
.PHONY: validate-reproducibility-mlflow
validate-reproducibility-mlflow:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Error: RUN_ID not provided. Usage: make validate-reproducibility-mlflow RUN_ID=<mlflow_run_id>"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) scripts/validate_reproducibility.py \
		--model-type random_forest \
		--reference-run-id $(RUN_ID) \
		--tolerance 1e-6

## Train pipeline with reproducibility
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/train_pipeline.py

## Train experiments with MLflow and reproducibility
.PHONY: train-experiments
train-experiments:
	$(PYTHON_INTERPRETER) src/experiments_mlflow.py


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
