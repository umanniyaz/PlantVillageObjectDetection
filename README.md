# Plant Village Disease Object Detection/Classification

Project Pre-requisites:
1. Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Creating an Environment(Virtual or Conda) with Python 3.8.12 and Installed Libraries from requirements.txt.Steps for setup are below:
Commands:
conda create --name env python==3.8.12
conda activate env
pip install -r requirements.txt

Project Objective: To Detect or classify whether disease is detected or not.

1. Plant Village dataset consists of healthy and disease images of leaves.There are 3 sets of healthy plant leaves i.e, Potato,Tomato and Pepperbell and 12 sets of plant disease leaves in the dataset.
2. Prepare a labelled dataset with Bounding Box Rectangles as labels(COCO format) in YOLO acceptible format.
3. After Dataset is prepared, train a YOLOv-8 Model and benchmark the model with validation set on perfomance metrics, Precision,recall,mAP.
4. Conversion of .pt model to .onnx format to maximize performance on hardware level with ONNX based Comptible Runtimes and portability across differenr frameworks.
5. Building and developing an Inference and using class based ROI detections for disease detection
6. Integration with basic User Interface provided by Streamlit


Project Directories:
1. models : Includes Object Detection Models in .pt and .onnx format
2. runs :  Consists of training results and graphs of metrics
3. config.json : Basic Configuration file
4. inference.py : Inference File for Detection of diseases in Plants
5. requirements.txt : All Project dependencies and libraries
6. clogging.py :  Logger File for debugging purposes
7. stream_lit.py : User Interface


Project Setup :

After Enviroment is setup with all dependencies,clone this repository and activate environment and run:

streamlit run stream_lit.py 


