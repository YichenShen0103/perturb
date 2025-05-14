# perturb

> This is a pytorch implementation of perturb, which add noise to images to attack id classifier while keep downstream task available

> [!WARNING]  
> For testing purposes only; accuracy is not guaranteed at this stage. 

## Repo Structure

```
perturb/
 |- data/
   |- faces/
   |- imgs/
   |- attr_celeba.csv
   |- attr_celeba_facenet.csv
 |- libs/
   |- BinaryClassifier.py
   |- CelebADataset.py
   |- DiT.py
 |- models/
   |- id_model.pth
   |- task_model.pth
   |- noise_model.pth
 |- samples/
   |- comparison.png
 |- util/
   |- data_preproccess.py
   |- evaluate.py
   |- train_facenet.py
 |- infer.py
 |- README.md ( This file )
 |- requirements.txt
 |- train.py
```

## Environment

Run commands below to prepare conda environment:

```shell
$ conda create -n perturb python=3.12
$ conda activate perturb
$ pip install -r requirements.txt
```

## Dataset

CelebA Download link: [Baidu NetDisk]() [Google Drive]()

### Data Preparation

Since we use facenet-pytorch, which need face to be right in the middle of the image, and the size of image should be 160 * 160, we use MTCNN to proccess the original images.

```shell
$ python utils/data_preparation.py 
```

## Pre-trained models

All pre-trained models or trained on CelebA dataset.

Download link: [Baidu NetDisk]() [Google Drive]()

## Train

Run commands below:

```shell
$ python train.py
```

If you want to train your own id or downstream task classifier, you shall delete the `.pth` file in `models/` respectively. If id classifier is to be trained, we provide you with a script `train_facenet.py` in `util/`.