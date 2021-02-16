

# 
SRC=src

# Django server
IMAGE=dmba
RUN=docker run -it --rm -v $(PWD):/code -v $(PWD)/$(SRC):/src $(IMAGE) 

bash:
	@ $(RUN) bash

tests:
	@ $(RUN) pytest -p no:cacheprovider

watch-tests:
	rm -f $(SRC)/.testmondata
	@ $(RUN) ptw --runner "pytest -o cache_dir=/tmp --testmon --quiet -rP"


# Docker container
build: touch-docker docker/image.dmba

touch-docker:
	touch docker/Dockerfile.dmba

docker/image.dmba: docker/Dockerfile.dmba
	docker build -t $(IMAGE) -f docker/Dockerfile.dmba .
	@ touch $@

