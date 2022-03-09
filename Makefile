

# 
SRC=src

# Django server
IMAGE=dmba
RUN=docker run -it --rm -v $(PWD):/code -v $(PWD)/$(SRC):/src $(IMAGE) 

bash:
	@ $(RUN) bash

tests:
	@ $(RUN) pytest -rP -p no:cacheprovider

watch-tests:
	rm -f $(SRC)/.testmondata
	@ $(RUN) ptw --runner "pytest -o cache_dir=/tmp --testmon --quiet -rP"

PYLINT_IMAGE=dmba-pylint
pylint:
	docker build -t $(PYLINT_IMAGE) -f docker/Dockerfile.pylint .
	@ docker run -it --rm -v $(PWD)/src:/src:ro $(PYLINT_IMAGE) dmba

freeze:
	docker run -it $(IMAGE) pip list --format=freeze | grep -v "^conda" > requirements.txt


# Docker container
build: images

images:
	docker build -t $(IMAGE) -f docker/Dockerfile.dmba .

