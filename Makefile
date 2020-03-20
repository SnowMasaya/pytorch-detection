help:
	@cat Makefile

DOCKER_FILE=./docker/Dockerfile
DOCKER_FILE_JETSON=./docker/Dockerfile.jetson
DOCKER_FILE_JETSON_ANNOTATION=./docker/Dockerfile.jetson.annotation
DOCKER=docker
DOCKER_JETSON=nvidia-docker
PYTHON_VERSION?=3.6
CUDA_VERSION?=10.2
CUDNN_VERSION?=7
SRC?=$(shell dirname `pwd`/pytorch_detection)
ANNOTATION_SRC?=$(shell echo "`pwd`/labelImg")
DISPLAY?=$(shell echo $DISPLAY)
ID=$(shell id -u)
WORK_DIR=/workspace/labelImg

build:
	docker build -t pytorch --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .
build-jetson:
	docker build -t pytorch-jetson --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE_JETSON) .
build-jetson-annotation:
	docker build -t annotation --build-arg python_version=$(PYTHON_VERSION) -f $(DOCKER_FILE_JETSON_ANNOTATION) .

bash:
	$(DOCKER) run --gpus=all --rm -it -v $(SRC):/workspace/pytorch_detection pytorch bash
bash-jetson:
	xhost + && $(DOCKER_JETSON) run -it --rm -v /usr/lib/python3.6/dist-packages/tensorrt:/usr/local/lib/python3.6/dist-packages/tensorrt -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/video0:/dev/video0:mwr -v $(SRC):/workspace/pytorch_detection pytorch-jetson bash
bash-annotation:
	$(DOCKER_JETSON) run --rm -it --user $(ID) -e DISPLAY=$(DISPLAY) --workdir=$(WORK_DIR) --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(ANNOTATION_SRC):/workspace/labelImg annotation make qt5py3;python3 ./labelImg/labelImg.py  
