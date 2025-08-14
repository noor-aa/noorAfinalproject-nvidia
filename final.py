
import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Paths to dataset
train_dir = "data/train"
val_dir  = "data/val"

net = jetson.inference.imageNet(
    model="models/gems_final_model/resnet18.onnx",
    labels="models/gems_final_model/labels.txt",
    input_blob="input_0",
    output_blob="output_0",
)
img = jetson.utils.loadImage("data/datas
et-expanded-splitt/test/Bixbite/bixbite_1.jpg")
class_id, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_id)
print(f"Predicted: {class_desc} (confidence: {confidence*100:.2f}%)")


img_np = jetson.utils.cudaToNumpy(img)

img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGBA2BGR)

label = f"{class_desc} ({confidence * 100:.1f}%)"
cv2.putText(img_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

cv2.imwrite("bixyy.jpg", img_bgr)

