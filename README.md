# OVERVIEW
Bone Fracture Detection done in Kaggle using Jupyter Notebook and Python. Evaluating a Faster R-CNN model from the library detectron2 for image detection.


# DATASET
The dataset includes:

- X-ray images categorized into multiple classes based on the fracture type.
- Annotations in COCO format for training, validation, and testing.
- Bounding boxes and segmentation masks for precise localization of fractures.
- Dataset Source:
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
   git clone https://github.com/NoorNick/Bone-Fracture-Detection.git
   cd Bone-Fracture-Detection
   ```

2. Install dependencies:
   ```bash
   pip install sys
   pip install os
   pip install torch
   pip install numpy
   pip install json
   pip install IPython
   pip install matplotlib
   pip install random
   pip install pickle
   ```

3. Install Detectron2:
   ```bash
   git clone https://github.com/facebookresearch/detectron2
   cd detectron2
   python setup.py build develop
   ```


# FEATURES
- Object detection and localization of fractures.
- Annotated dataset with bounding boxes and segmentation masks.
- Custom Faster R-CNN configuration for optimized performance.
- Visualization of predictions and training metrics.

# PERFORMANCE METRICS (Faster R-CNN with ResNet-50 FPN)

- Mean Average Precision (mAP): 0.916
- Class Accuracy: 0.966
- Loss: 0.262
- Loss Box Regression: 0.1165

# PERFORMANCE METRICS (Faster R-CNN with ResNeXt-101 FPN)

- Mean Average Precision (mAP): 0.901
- Class Accuracy: 0.962
- Loss: 0.261
- Loss Box Regression: 0.1134


