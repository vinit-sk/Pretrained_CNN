# Pretrained_CNN
Cat-Dog Classification model trained on resent50 architecture. 

### Classification using Pytorch

This is a common computer vision project of image classification. Here we have trained resnet50 model. While the output is the accuracy, the main objective of this project is not to get a high accuracy but rather to learn how to use convolution neural network (CNN) for classification using Pytorch.

### Dataset Link
https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

### Installations

Create a new environment and run the following command in your terminal
```

pip install -r requirement.txt

```
### data split

if data is not in data pipeline format, provide dataset path and run 
```
python split_data.py 
```
it will create dataset in data pipeline format.

### data pipeline
![data_pipeline (3)](https://github.com/user-attachments/assets/30d82ef0-2c4b-434d-b18d-755b09455719)

### Usage

To execute the project, you only need to run the file below in your terminal

```
python train_pretrained_resnet50_model.py
```

