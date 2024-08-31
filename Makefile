.PHONY: install

SHELL := /bin/bash

NVCC = /usr/local/cuda-11.3/bin/nvcc
PYENV_ROOT = $(HOME)/.pyenv
PYTHON_VERSION ?= 3.12.2
VENV = .venv
PYTHON = ./$(VENV)/bin/python3

# echo -e '\nif [ -d "/usr/local/cuda-11.3/bin/" ]; then' >> ~/.profile
# echo -e "    export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}" >> ~/.profile
# echo -e "    export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.profile
# echo "fi" >> ~/.profile

cuda_export:

	@export PATH=/usr/local/cuda-11.3/bin:$(PATH)
	@export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$(LD_LIBRARY_PATH)

pyenv: cuda_export
	@export PYENV_ROOT=$(PYENV_ROOT)
	@bash ./setup_pyenv.sh
	@export PATH=$(PYENV_ROOT)/bin:$(PATH)
	@source ~/.bashrc
	@eval "$(pyenv init -)"
	@pyenv install -s $(PYTHON_VERSION)
	@pyenv global $(PYTHON_VERSION)

venv/bin/activate: pyenv 
	@source ~/.bashrc
	@echo "Using $(shell python -V)"
	@python3 -m venv $(VENV)
	@chmod +x $(VENV)/bin/activate
	@source ./$(VENV)/bin/activate

venv: venv/bin/activate
	@source ./$(VENV)/bin/activate
	@echo "VIRTUAL ENVIRONMENT LOADED"

install: venv