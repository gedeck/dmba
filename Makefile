SRC=src
DMBA_BASE=dmba-base
DMBA_DEV=dmba-dev
DMBA_JUPYTER=dmba-jupyter


bash:
	docker run -it --rm -v $(PWD):/code $(DMBA_BASE) bash

jupyter:
	docker run --rm -v $(PWD):/code -p 8931:8931 $(DMBA_JUPYTER) jupyter-lab --allow-root --port=8931 --ip 0.0.0.0 --no-browser


tests:
	docker run -it --rm -v $(PWD):/code $(DMBA_DEV) pytest src

watch-tests:
	rm -f .testmondata
	docker run -it --rm -v $(PWD):/code $(DMBA_DEV) ptw --runner "pytest --testmon"
	rm -f .testmondata

isort:
	docker run -it --rm -v $(PWD):/code $(DMBA_DEV) isort src

ruff:
	docker run -it --rm -v $(PWD):/code $(DMBA_DEV) ruff --fix .

mypy:
	docker run -it --rm -v $(PWD):/code $(DMBA_DEV) mypy src


# Docker container
images:
	docker build -t $(DMBA_BASE) -f docker/Dockerfile.base .
	docker build -t $(DMBA_DEV) -f docker/Dockerfile.devtools .
	docker build -t $(DMBA_JUPYTER) -f docker/Dockerfile.jupyter .

# Virtualenv
venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
