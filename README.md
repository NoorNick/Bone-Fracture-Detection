## Overview
Bone Fracture Detection done in Kaggle using Jupyter Notebook and Python. Evaluating a Faster R-CNN model from the library detectron2 for image detection.


## Dataset
The dataset includes:

- X-ray images categorized into multiple classes based on the fracture type.
- Annotations in COCO format for training, validation, and testing.
- Bounding boxes and segmentation masks for precise localization of fractures.
- Dataset Source:
https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project

## models used
From the library detectron2, 2 models were extracted Faster R-CNN with ResNeXt-101 FPN and Faster R-CNN with ResNet-50 FPN

## Libraries needed
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


## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/NoorNick/Bone-Fracture-Detection.git
   cd Bone-Fracture-Detection
   ```

2. Install dependencies:
  
   ```bash
   pip install -r requirements.txt
   ```

3. Install Detectron2:
   ```bash
   git clone https://github.com/facebookresearch/detectron2
   cd detectron2
   python setup.py build develop
   ```


## Features
- Object detection and localization of fractures.
- Annotated dataset with bounding boxes and segmentation masks.
- Custom Faster R-CNN configuration for optimized performance.
- Visualization of predictions and training metrics.

## Performance metrics (Faster R-CNN with ResNet-50 FPN)

- Mean Average Precision (mAP): 0.916
- Class Accuracy: 0.966
- Loss: 0.262
- Loss Box Regression: 0.1165

## Performance metrics (Faster R-CNN with ResNeXt-101 FPN)

- Mean Average Precision (mAP): 0.901
- Class Accuracy: 0.962
- Loss: 0.261
- Loss Box Regression: 0.1134

## Contributing

We welcome contributions! To contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to your forked repository (`git push origin feature-branch`).
6. Create a pull request.

Please ensure that your code follows the existing style and that tests are included for new functionality.

## License

This project is created for educational purposes as part of a university course. All rights reserved to the author. You may not use, modify, or distribute this project for commercial purposes without permission from the author. If you wish to use this project for educational or research purposes, please contact the author for further permission.

## Author
Noor Nick


