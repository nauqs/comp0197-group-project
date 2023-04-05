# Applied Deep Learning (COMP0197) group project

## Semi-supervised segmentation using deep neural networks

Implement and evaluate a semi-supervised neural network for semantic segmentation of images on the [The Oxford-IIIT Pet Data-set](https://www.robots.ox.ac.uk/~vgg/data/pets/).


### Instructions

1. Create environment

```
conda create --name comp0197-group-project -c pytorch python=3.10 pytorch=1.13 torchvision=0.14
conda activate comp0197-group-project
```

2. Install required packages

```
pip install -r requirements.txt
```

3. Training

To train in Colab:
- clone this repository into your Google Drive account
- create a new Google Colab notebook
- in the Colab menu bar, click Runtime -> Change runtime type -> Hardware accelerator -> GPU
- go to files, click Mount Google Drive
- change your working directory to this repository 
    (using ```os.chdir``` or ```%cd path/to/comp0197-group-project/```)
- to get weights run from hf, run:
```snapshot_download(repo_id="nauqs-ucl/comp0197-group-project", local_dir="content/", local_dir_use_symlinks="auto")```
    this will download everything to the ```content/``` dir on Colab
- to install the dependencies, you can run the following:
    ```
    !pip install -q condacolab
    import condacolab
    condacolab.install()
    !conda install -c  pytorch python=3.9 pytorch=1.13 torchvision=0.14
    ```
- run ```! python train_single_model.py``` with required flags:
  - The default model is the supervised one with no semi-supervised learning.
  - Use ```--pseudo-label``` for semi-supervised learning
  - Use ```--cutmix``` for cutmix augmentation, there is none by default 
  - Use ```--cutout``` for cutout augmentation, there is none by default 
  - Use ```--large-resnet``` for Resnet101, Resnet50 is used by default
  - Use ```--no-pretrained``` for no pretrained Resnet weights, we are loading them by default

  - Run ```! python main.py -h``` for more information on the available flags.
