# Perturb

> This is a pytorch implementation of perturb, which add noise to images to attack id classifier while keep downstream task available

![comparison](assets/comparison.png)

> [!TIP]  
> Run `demo.py` to view a comparison between the noise-added images and the original images like above.

## ‼️ Latest Performance

> **🌟 May 18**    
> New version! It's now extremely hard to distinguish perturbed images and original ones. 

|               | Original | Perturbed  |
| ------------- | -------- | ---------- |
| ID Accuracy   | 96.16%   | **46.81%** |
| Task Accuracy | 98.24%   | **98.64%** |

You can run `evaluate.py` to analysis your local models.


## Repo Structure

```
perturb/
 |- assets/
   |- comparison.png
 |- data/
   |- faces/
   |- imgs/
   |- attr_celeba.csv
   |- attr_celeba_facenet.csv
 |- libs/
   |- BinaryClassifier.py
   |- CelebADataset.py
   |- DiT.py
   |- SSIMLoss.py
 |- models/
   |- id_model.pth
   |- task_model.pth
   |- noise_model.pth
 |- util/
   |- data_preparation.py
   |- evaluate_models.py
 |- demo.py
 |- evaluate.py
 |- README.md ( This file )
 |- requirements.txt
 |- train_facenet.py
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

CelebA Download link: [Baidu NetDisk](https://pan.baidu.com/s/1rCKjFZhh5IzwZnfdUFl7lA?pwd=w37x) [Google Drive](https://drive.google.com/file/d/1Tn_w3Kg2yMMZnfJ_khp0PR9WbGdF_i6O/view?usp=share_link)

### Data Preparation

Since we use facenet-pytorch, which need face to be right in the middle of the image, and the size of image should be 160 * 160, we use MTCNN to proccess the original images.

```shell
$ python utils/data_preparation.py 
```

## Pre-trained models

All pre-trained models were trained on CelebA dataset.

Download link: [Baidu NetDisk](https://pan.baidu.com/s/1FnhqS5mhIBoGSYjzr4hqvg?pwd=134b) [Google Drive](https://drive.google.com/drive/folders/1Ygomk9mUZmaEDaA_NPkLy_QqMQu08W-3?usp=share_link)

## Train

Run commands below:

```shell
$ python train.py
```

If you want to train your own id or downstream task classifier, you shall delete the `.pth` file in `models/` respectively. If id classifier is to be trained, we provide you with a script `train_facenet.py`.