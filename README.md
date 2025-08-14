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

Set path to unzipped dataset
input_folder = '/home/noor/jetson-inference/python/training/classification/data/dataset-expanded'
output_folder = '/home/noor/jetson-inference/python/training/classification/data/dataset-expanded-splitt'


splitfolders.ratio(input_folder,
                   output=output_folder,
                   seed=42,
                   ratio=(.9, .05, .05),
                   group_prefix=None)
) 
this code split it into test, train, val. 
the labels were made through this as well as setting up the onnx in the docker. i changed the labels to classify not only the title of the gem but also if it is precious, semi precious collectors or organic. i used the abbreviations P, SP, COLL and ORG. 

# to code the classifier the code looked like this: 

import jetson.inference
import jetson.utils
import cv2
import numpy as np


train_dir = "data/train"
val_dir  = "data/val"

net = jetson.inference.imageNet(
    model="models/gems_final_model/resnet18.onnx",
    labels="models/gems_final_model/labels.txt",
    input_blob="input_0",
    output_blob="output_0",
)
img = jetson.utils.loadImage("data/dataset-expanded-splitt/test/Bixbite/bixbite_1.jpg")
class_id, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_id)
print(f"Predicted: {class_desc} (confidence: {confidence*100:.2f}%)")


img_np = jetson.utils.cudaToNumpy(img)

img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGBA2BGR)

label = f"{class_desc} ({confidence * 100:.1f}%)"
cv2.putText(img_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

cv2.imwrite("bixyy.jpg", img_bgr)

# video link
https://drive.google.com/file/d/1ZezxptxTfTddLw87IJTCiepcJc7zwQLA/view?usp=sharing 
