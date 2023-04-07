# Applied Deep Learning (COMP0197) group project

## Semi-supervised segmentation using deep neural networks

Implement and evaluate a semi-supervised neural network for semantic segmentation of images on the [The Oxford-IIIT Pet Data-set](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Structure
```datasets.py``` contains our data handling  
```models.py``` contains our deeplabv3 model  
```utils.py``` contains our code for image augmentation and prediction visualisation  
```train.py``` contains training code for supervised and semi-supervised models  
```run.py``` runs all the experiments to reproduce our results


### Instructions

1. Create environment  

```
conda create --name comp0197-group-project -c pytorch python=3.10 pytorch=1.13 torchvision=0.14
conda activate comp0197-group-project
```

2. Install required packages  
This only installs wandb currently.

```
pip install -r requirements.txt
```

3. Training  
We have logged our metrics using WandB. If you would like to log the results in your own WandB project, please edit the ```wandb.init()``` command in ```run.py``` and run ```python run.py``` to reproduce our experiments.  
To run without WandB and get the metrics printed in the terminal, run ```WANDB_MODE=disabled python run.py ```. We recommend piping the stdout to a file due to the number of lines outputted.
