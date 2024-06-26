#!/usr/bin/env python3
"""
This is the yolo project
By Ced
"""
from keras.models import load_model
import numpy as np
import math

class Yolo():
    """
    Initialize Yolo
    implementing model with load_model
    an class name with read_classes(self,path)
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        self.class_names = self.read_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def read_classes(self, path):
        """
        function that prepare the label classes
        """

        list_class = []
        file = open(path, mode='r')
        classes = file.read().split("\n")
        for i in classes[0:-1]:
            list_class.append(i)

        return list_class

    def process_outputs(self, outputs, image_size):
        # useless for now
        grid_height, grid_width, anchor_boxes, lastclasse = outputs[0].shape
        image_height, image_width = image_size[0], image_size[1]
        # print("test", outputs[0][0,0,0,:])
        box_confidence=[]
        boxes=[]
        box_class_probs=[]
        print("tete", outputs[0][:, :, :, 0:4])
        for output in outputs:
                box_confidence.append( 1 / ( 1 + np.exp(-output[:, :, :, 4])))
                box_class_probs.append(1 / ( 1 + np.exp(-output[:, :, :, 5:])))
        
        return boxes, box_confidence, box_class_probs

    """
    def process_outputs(self, outputs, image_size):
       
        split outputs into
        boxes, box confidence and box class prob
       

       

        #(t_x, t_y, t_w, t_h) = outputs[:, :, :, 0:4]
        t = outputs[0][:, :, :, 0]
        t_x = outputs[0][:, :, :, 1]
        t_y = outputs[0][:, :, :, 2]
        t_h = outputs[0][:, :, :, 3]
        #box_confidence = outputs[0][:, :, :, 4]
        #(x1, y1, x2, y2) = (image_width * t_x, image_height * t_y, image_width * t_x + image_width * t_w, image_height * t_y + image_height * t_h   )    
        boxes = []
        #print("boxes",Pc)
        box_confidence = outputs[0][:, :, :, 4]
        box_class_probs = outputs[0][:, :, :, 5:]
        #print("boxes prob", box_class_probs.shape)

        return boxes, box_confidence, box_class_probs
    """


# Boxes: [array([[[[-2.13743365e+02, -4.85478868e+02,  3.05682061e+02,
#            5.31534670e+02],
