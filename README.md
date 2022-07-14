# robocup2022-cv-what-is-that-yolov5

## Installation

1. Clone this project
2. Create conda environment
```
$ conda create -n wit python=3.9
```
then
```
``` bash
$ conda install --yes pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements_localtest.txt
```

## Weight
You can change weight.pt by put a new weight in project folder and change a parameter in yolov5.py
- WEIGHTS_PATH = 'custom_object_weight.pt'


## Weight
You can change data.yaml by put a new .yaml in data folder and change a parameter in yolov5.py
- data=ROOT / 'data/data.yaml'
