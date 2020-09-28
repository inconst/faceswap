# Faceswap

PyTorch implementation of face swapping approach described in paper: 
[FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping](https://arxiv.org/pdf/1912.13457.pdf).

This repo is inspired by [taotaonice/FaceShifter](https://github.com/taotaonice/FaceShifter) and
uses code from [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) 
(an implementation of [arcface](https://arxiv.org/abs/1801.07698) paper)

------

## Prerequisites


You can create [conda](https://docs.conda.io/en/latest/miniconda.html) environment with all necessary 
dependencies by running (tested on Ubuntu 18.04):
```bash
conda create --name myenv python=3.7
conda activate myenv
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install opencv-python tqdm tensorboardX 
```

## Running pretrained model

1. Download arcface model (`model_ir_se_50.pth`) and pretrained weights of our 
model (`G_latest.pth` and `D_latest.pth`) 
from [Google Drive](https://drive.google.com/drive/folders/10FDMU0tV5zn39nJ73j6_QJhJ7odi1Mz5?usp=sharing) 
and place them in `saved_models` folder

2. Run test script:
    ```
    python test.py
    ```
    Here you should choose source and target images with faces 
    (face will be detected and cropped to match expected model input) and then perform face swapping.
    
    Directory `data/test_images` contains images to try out face swapping

## Training model from scratch


1. Download arcface model (`model_ir_se_50.pth`) 
from [Google Drive](https://drive.google.com/drive/folders/10FDMU0tV5zn39nJ73j6_QJhJ7odi1Mz5?usp=sharing) 
and place it in `saved_models` folder

2. [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) (70000 face images) is used for training. 
You should download `thumbnails128x128.zip` (~2GB) from 
[Google Drive](https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS)
and unzip it in `data` folder

3. Run training script:
    ```
    python train.py
    ```
    You can run `python train.py --help` to see all options with additional info.
    
    Latest models will be saved in `saved_models` directory. 
    
    Statistics about training losses is saved in `tensorboard_stats` directory 
    for viewing in [Tensorboard](https://www.tensorflow.org/tensorboard)
   

