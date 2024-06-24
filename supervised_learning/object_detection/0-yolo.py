#!/usr/bin/env python3
from keras.models import load_model
import numpy as np


class Yolo():
    #global list_class
    
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model= load_model(model_path)
        self.class_names=self.read_classes(classes_path)
        self.class_t=class_t
        self.nms_t=nms_t
        self.anchors=anchors
    


    def read_classes(self,path):
        
        list_class=[]
        file= open(path, mode='r')
        classes = file.read().split("\n")
        for i in classes:
            list_class.append(i)        
        
        return list_class
