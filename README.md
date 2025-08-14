# noorAfinalproject-nvidia
# Project Name
GEMSTONECLASS-4068

This project uses ResNet18, a convolutional neural network originally trained on ImageNet, and fine-tuned for a custom gemstone classification dataset.

Model Architecture: ResNet18 (ONNX format)
Training Framework: PyTorch
Deployment: NVIDIA orin nano

1. The model receives an input image of a gemstone.
2. Image is processed and resized to match model input.
3. The model predicts a class label from the fine-tuned gemstone classes 
4. The prediction of class,precious-level and confidence score are displayed.


The resnet18 was retrained based on a dataset imported from kaggle, the dataset had a collection of images classified under a folder named by the gemstone. The dataset was split using a python code 
(import splitfolders


# Set path to unzipped dataset
input_folder = '/home/noor/jetson-inference/python/training/classification/data/dataset-expanded'
output_folder = '/home/noor/jetson-inference/python/training/classification/data/dataset-expanded-splitt'


splitfolders.ratio(input_folder,
                   output=output_folder,
                   seed=42,
                   ratio=(.9, .05, .05),
                   group_prefix=None)
) 
into test, train, val. 

insert video link:
