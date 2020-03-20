# pytorch-detectnet
Training of object detection networks with PyTorch

# ENvironiments

- Ubuntu 18.04
- Jetson Nano 

# Prerequrements for Jetson case

## Checking Jetson Performance


[jetson_status](https://github.com/rbonghi/jetson_stats)

```
sudo apt-get install python3-pip
sudo -H pip3 install jetson-stats
```

After install of it, You can check the performance by the following command

```
sudo jtop
```

## Jetson Clocks for improve performance

```
sudo jetson_clocks
```

## Setup Swap Memory for large memory process

```
git clone https://github.com/JetsonHacksNano/installSwapfile
cd installSwapfile
./installSwapfile.sh
sudo reboot now
```

## Change default docker process for using gpu


```
sudo vim /etc/docker/daemon.json 
```

You add a `default-runtime`

```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

```
service docker restart
```
# Getting data

[FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/index.html#download)

Image data
- http://tamaraberg.com/faceDataset/originalPics.tar.gz
Annotation Data
- http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz


# How to prepare set up environments

## Get data and code for Jetson case

```
mv {Download Directory}/FDDB-folds.tgz .
tar -xvzf FDDB-folds.tgz
mkdir originalPics
cd originalPics
mv {Download Directory}/originalPics.tar.gz .
tar - xvzf originalPics.tar.gz
cp ../FDDB-folds/*.txt  .
```

Fix `train_jetson.py` your setting training data path

```py
    train_dataset = FaceDataset(
        root_dir='{set up your data set folder}/originalPics',
        fold_dir='{set up your data set folder}/FDDB-folds',
        fold_range=range(1,11),
        transform=transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            #transforms.RandomResizedCrop(args.resolution),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = FaceDataset(
        root_dir='{set up your data set folder}/originalPics',
        fold_dir='{set up your data set folder}/FDDB-folds',
        fold_range=range(1,11),
        transform=transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            #transforms.Resize(256),
            #transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ]))
```

# How to train for Jetson case 

```
make build-jetson
make bash-jetson

python train_jetson.py
```

# How to predict for Jetson case

Fix `inference.py` your setting inference data path

```py
        if args.data == "FaceDataset":
            inference_dataset = FaceDataset(
                root_dir='datasets/raw/originalPics',
                fold_dir='datasets/raw/originalPics',
                fold_range=range(1,11),
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )
```

```
make build-jetson
make bash-jetson

python inference.py
```

You can confirm the predict data `predict_image` folder.
This folder is automatic made by script

## Run the inference code camera mode for Jetson case

```
make build-jetson
make bash-jetson
python inference.py --camera
```


# If you make up own data set for Jetson case

labelImage
- https://github.com/tzutalin/labelImg


Working directory `pytorch-detection-share_version2/`

```
git clone https://github.com/tzutalin/labelImg
make build-jetson-annotation
make bash-annotation
```

If you fail this command, you have to check the file permission

## Run the train code for Ubutu case

Fix `train_jetson.py` set training data path

```
        train_dataset = LabelImage(
            data_dir='test_image',
            transform=transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = LabelImage(
            data_dir='test_image',
            transform=transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                normalize,
            ]))
```

```
make build-jetson
make bash-jetson

python train_jetson.py --data LabelImg
```

## Run the inference code 

Fix `inference.py` your setting inference data path

```py
        elif args.data == "LabelImg":
            inference_dataset = LabelImage(
                data_dir='./test_image/',
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )

```

```
make build-jetson
make bash-jetson

python inference.py --data LabelImg
```

You can confirm the predict data `predict_image` folder.
This folder is automatic made by script

# Prerequrements for Ubuntu case

## Docker for Ubuntu case
- https://docs.docker.com/install/linux/docker-ce/ubuntu/

With-out `sudo`

```
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo systemctl restart docker
```

re-logiun

## cuda and graphics drivers install for Ubuntu case

- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation

## cuDNN install for Ubuntu case

Download
- https://developer.nvidia.com/rdp/cudnn-download

Install instruction
- https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

## TensorRT install for Ubuntu case

Download
- https://developer.nvidia.com/nvidia-tensorrt-7x-download

Install instruction
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html


## Docker GPU setting for Ubuntu case

- https://github.com/NVIDIA/nvidia-docker/tree/master#quickstart

# Authorization for Ubuntu case

Getting NVIDIA GPU CLOUD API

https://ngc.nvidia.com/setup/api-key

```
docker login nvcr.io 
Username: $oauthtoken 
Password: {Generate API Key}
```

# Getting PyTorch Docker for Ubutu case

```
docker pull nvcr.io/nvidia/pytorch:20.02-py3
```

# Getting data

[FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/index.html#download)

Image data
- http://tamaraberg.com/faceDataset/originalPics.tar.gz
Annotation Data
- http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz


# How to prepare set up environments

## Get data and code for Ubutu case

```
mv {Download Directory}/FDDB-folds.tgz .
tar -xvzf FDDB-folds.tgz
mkdir originalPics
cd originalPics
mv {Download Directory}/originalPics.tar.gz .
tar - xvzf originalPics.tar.gz
cp ../FDDB-folds/*.txt  .
```

Fix `train.py` your setting training data path

```py
    train_dataset = FaceDataset(
        root_dir='{set up your data set folder}/originalPics',
        fold_dir='{set up your data set folder}/FDDB-folds',
        fold_range=range(1,11),
        transform=transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            #transforms.RandomResizedCrop(args.resolution),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = FaceDataset(
        root_dir='{set up your data set folder}/originalPics',
        fold_dir='{set up your data set folder}/FDDB-folds',
        fold_range=range(1,11),
        transform=transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            #transforms.Resize(256),
            #transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ]))
```

# How to train for Ubutu case 

```
make build
make bash

python train.py
```

# How to fast train by AMP for Ubutu case 

Requirements: GPU with Tensor Core implemented such as a Volta and so on

```
make build
make bash

python train.py --amp
```

# How to predict for Ubutu case

Fix `inference.py` your setting inference data path

```py
        if args.data == "FaceDataset":
            inference_dataset = FaceDataset(
                root_dir='datasets/raw/originalPics',
                fold_dir='datasets/raw/originalPics',
                fold_range=range(1,11),
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )
```

```
make build
make bash

python inference.py
```

You can confirm the predict data `predict_image` folder.
This folder is automatic made by script

# Export ONNX model 

```
make build
make bash

python3 onnx_export.py 
```

# Export ONNX model

```
make build
make bash

python3 onnx_validate.py {your onnx model for example "resnet18.onnx"}
```

# If you make up own data set for Ubutu case

labelImage
- https://github.com/tzutalin/labelImg

```
git clone https://github.com/tzutalin/labelImg
cd labelImg/
docker run -it --user $(id -u) -e DISPLAY=unix$DISPLAY --workdir=$(pwd) --volume="/home/$USER/labelImg:/home/$USER/labelImg" --volume="/home/$USER/pytorch-detection:/home/$USER/pytorch-detection" --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" -v /tmp/.X11-unix:/tmp/.X11-unix tzutalin/py2qt4
make qt4py2;./labelImg.py
```

## Run the train code for Ubutu case

Fix `train.py` set training data path

```
        train_dataset = LabelImage(
            data_dir='test_image',
            transform=transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = LabelImage(
            data_dir='test_image',
            transform=transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                normalize,
            ]))
```

```
make build
make bash

python train.py --data LabelImg
```

## Run the inference code 

Fix `inference.py` your setting inference data path

```py
        elif args.data == "LabelImg":
            inference_dataset = LabelImage(
                data_dir='./test_image/',
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )

```

```
make build
make bash

python inference.py --data LabelImg
```

You can confirm the predict data `predict_image` folder.
This folder is automatic made by script
