
import logging
import os
import time
import warnings
from datetime import datetime, timedelta
import onnxruntime
import json
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch



from clogging import LOGGING_PATH, setup_logger


time_run = datetime.now()  
dt_string = time_run.strftime("%d-%m-%Y")
log_fname = '%s' % dt_string
logger = setup_logger(log_fname,LOGGING_PATH +
                       '/log_%s.log' % dt_string, level=logging.DEBUG)

                       
class ObjectDetectionClassification:
    """
        Performs the Object Localization or extraction into important entities of poses from image and gives coorniates for detected bbox
        and also gives the classification if images into 2 classes Helathy,Disease
    
    Args:
        self: constructor,
        combined_json: configuration file json

    Returns: 
        None
    
    """

    def __init__(self,img):
        self.file_dir = os.getcwd()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(['FILE DIR',self.file_dir])
        self.img = img
        #Config file
        with open("config.json") as config_file:
            self.config_data = json.load(config_file)
        config_file.close()
        self.model_path = os.path.join(self.file_dir,self.config_data['models_location_path'])
        self.object_detection_model = os.path.join(self.model_path,self.config_data['obj_detection_model'])

    def load_model(self):
        """
            Loads Object Detection Model in ONNX runtime.

        Args:
            self: constructor
        
        Returns:
            ONNX Runtime inference session
        """
        try:           
            cuda = False
            model_name = self.object_detection_model
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']         
            session = onnxruntime.InferenceSession(model_name, providers=providers)
            return session
        except Exception as e:
            print(['Exception occured in loading model',e])

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        '''
            Rescale coords (xyxy) from img1_shape to img0_shape
            Args:
            img1_shape: resized image shape
            img0_shape: original input image shape
            coords: predictions returned from object detection model

            Returns:
            ROI coordinates in original image
        '''
        try:
            if ratio_pad is None:  # calculate from img0_shape
                gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
                pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            coords[ [0, 2]] -= pad[0]  # x padding
            coords[ [1, 3]] -= pad[1]  # y padding
            coords[ 0:4] /= gain
            return coords
        except Exception as e:
            print(['Exception occured in scale_coords function ',e])

    def letterbox(self,im: np.array, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """
            Creates Letter Box or gives depth,width,height of boxes.

        Args:
            self: constructor,
            im (array): image of input card,
            new_shape (filter): image resizing filter,
            color:  color in RGB channel,
            auto,scaleup,stride: other tuning parameters

        Returns:
            Image with ratio,width and height of boxes
        """
        try:
            # Resize and pad image while meeting stride-multiple constraints
            shape = im.shape[:2]  # current shape [height, width]
            
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better val mAP)
                r = min(r, 1.0)

            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_CUBIC)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            # The image returned below gets degraded
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            return im, r, (dw, dh)
        except Exception as e:
            print(['Exception occured in creating bounding box',e])


    def preprocess_image(self):
        """
            Preprocessing of input image.

        Args:
            self: constructor,
            img (array): image of input 

        Returns:
            Image with ratio,width and height of boxes
        """
        try:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logger.info(['Processing Images'])
            image = self.img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            original_image = image
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            im /= 255
            return im,original_image,ratio,dwdh
        except Exception as e:
                print(['Exception occured in preprocessing image',e])



    def object_model_predictions(self,img,ort_session):
        """
            Object Detection Model predictions using ONNX runtime.

        Args:
            self: constructor,
            img (array): image of input ,
            ort_session: onnx modelsession,

        Returns:
            Output predictions of different rois or entities with craft coordinates
        """
       
        try:
            outname = [i.name for i in ort_session.get_outputs()]
            inname = [i.name for i in ort_session.get_inputs()]     
            inp = {inname[0]:img}          
            # ONNX inference
            outputs = ort_session.run(outname, inp)[0]
            return outputs
        except Exception as e:
            print(['Exception occured in ONNX Inference session and predictions',e])


    def roi_extraction(self,img,outputs,im0): 
        """
            Extracts ROIs from the Model outputs

        Args:
            self: constructor,
            img (array): preprocessed image,
            original_image: original input image ,
            ratio (float):  ratio, in which boxes are proportioned,
            dw,dh (float): width and height of bounding box,
            outputs (array): bounding boxes predictions for different entities
            im0: original input image

        Returns:
            Bounding box extracted for different entities clubbed in a Dictionary
        """

        try:
            # roi ext
            roi_dict = dict()
            pred = outputs
            names = ['disease','healthy']

            # Process detections
            for det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det1 = self.scale_coords(img.shape[2:], det[1][1:5], im0.shape).round()
                    logger.info(['Scaled Coordinates',det1])
                    cls = det[1][5]
                    score = det[1][6]
                    cls_id = int(cls)
                    name = names[cls_id]
                    logger.info(['Class Name',name])
                    score = round(float(score),3)
                    box_coordinates = [int(coord) for coord in det1]
                    logger.info(['Confidence Score',score])
                    im = cv2.rectangle(im0, (box_coordinates[0], box_coordinates[1]), (box_coordinates[2], box_coordinates[3]), (0, 255, 0), 2)
                    # Add text to the image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im, name, (box_coordinates[0], box_coordinates[1] - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return im
        except Exception as e:
            print(['Exception occured in roi_extraction from images',e])


    def main(self):
        """
            Integration of all above methods,gives coordinates of bbox and image classification results

        Args:
            self: constructor,
            image (array): original input image
        
        Returns:
            Dictionary with bbox img
        """
        try:
            logger.info(['Starting Main Function In Object Detection/Classifiation'])
            start_t = time.perf_counter()
            object_model = self.load_model()
            img,_,_,_ = self.preprocess_image() 
            preds = self.object_model_predictions(img,object_model)
            im = self.roi_extraction(img,preds,self.img)
            end_t = time.perf_counter()
            print("Time Taken for execution:",end_t-start_t) 
            logger.info(["Returning Output Image"]) 
            return im
        except Exception as e:
            logger.debug(['Exception occured in main function',e])
