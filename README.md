# OVERVIEW
Bone Fracture Detection done in Kaggle using Jupyter Notebook and Python. Evaluating a Faster R-CNN model from the library detectron2 for image detection.

# DATASET
The dataset includes:

-X-ray images categorized into multiple classes based on the fracture type.
-Annotations in COCO format for training, validation, and testing.
-Bounding boxes and segmentation masks for precise localization of fractures.
Dataset Source:
https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project

# MODEL USED
From the library detectron2, 2 models were extracted Faster R-CNN with ResNeXt-101 FPN and Faster R-CNN with ResNet-50 FPN

# LIBRARIES NEEDED
- sys
- os
- distutils.core
- torch
- detectron2
- numpy
- json
- cv2
- IPython
- PIL
- matplotlib
- random
- pickle

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bone-fracture-detection.git
   cd bone-fracture-detection
   ```

2. Install dependencies:
   ```bash
   pip install pyyaml==5.1
   pip install -r requirements.txt
   ```

3. Install Detectron2:
   ```bash
   git clone https://github.com/facebookresearch/detectron2
   cd detectron2
   python setup.py build develop
   ```

---

## **Usage**

### 1. Data Preparation
Register the COCO dataset with Detectron2:
```python
from detectron2.data.datasets import register_coco_instances

register_coco_instances("bone_fractures_train", {}, "path_to_train_annotations.json", "path_to_train_images")
```

### 2. Training
Train the Faster R-CNN model:
```bash
python train.py --config-file configs/faster_rcnn_R_50_FPN.yaml --num-gpus 1
```

### 3. Evaluation
Evaluate the trained model on the validation dataset:
```bash
python evaluate.py --model-path output/model_final.pth
```

### 4. Visualizations
Generate random visualizations from the training dataset:
```bash
python visualize.py --dataset-name bone_fractures_train --num-images 5
```

---

# FEATURES
-Object detection and localization of fractures.
-Annotated dataset with bounding boxes and segmentation masks.
-Custom Faster R-CNN configuration for optimized performance.
-Visualization of predictions and training metrics.

# PERFORMANCE METRICS

## Faster R-CNN with ResNet-50 FPN
-Mean Average Precision (mAP): 0.916
-Class Accuracy: 0.966
-Loss: 0.262
-Loss Box Regression: 0.1165

## Faster R-CNN with ResNeXt-101 FPN
-Mean Average Precision (mAP): 0.901
-Class Accuracy: 0.962
-Loss: 0.261
-Loss Box Regression: 0.1134

## **References**

1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. [arXiv](https://arxiv.org/abs/1506.01497)  
2. [Detectron2 Documentation](https://detectron2.readthedocs.io/en/latest/)  
3. [Colab Notebook Example](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)  
4. [Bone Fracture Dataset](https://universe.roboflow.com/veda/bone-fracture-detection-daoon)  


