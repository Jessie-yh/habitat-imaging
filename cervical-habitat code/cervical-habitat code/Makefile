PYTHON ?= python

.PHONY: help install dce ivim lint clean

help:
	@echo "install   install the package"
	@echo "dce       run the DCE k=3 configuration"
	@echo "ivim      run the IVIM k=4 configuration"
	@echo "lint      ruff check"

install:
	$(PYTHON) -m pip install -e .

dce:
	$(PYTHON) -m cervical_habitat run -c configs/dce_rates_k3.yaml

ivim:
	$(PYTHON) -m cervical_habitat run -c configs/ivim_d_dstar_f_k4.yaml

lint:
	$(PYTHON) -m ruff check src

clean:
	rm -rf build dist src/*.egg-info
