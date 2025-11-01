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

## Start MLflow UI server
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --port 5000

## List MLflow experiments
.PHONY: mlflow-list
mlflow-list:
	mlflow experiments list


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
