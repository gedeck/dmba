SRC=src
DMBA_BASE=dmba-base
DMBA_DEV=dmba-dev


bash:
	docker run -it --rm -v $(PWD):/code $(DMBA_BASE) bash

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

# Virtualenv
venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
