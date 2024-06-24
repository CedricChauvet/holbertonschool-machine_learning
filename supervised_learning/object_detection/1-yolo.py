#!/usr/bin/env python3
"""
This is the yolo project
By Ced
"""
from keras.models import load_model
import numpy as np


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
        """
        split outputs into
        boxes, box confidence and box class prob
        """

        # useless for now
        grid_height, grid_width, anchor_boxes, lastclasse = outputs[0].shape
        image_height, image_width = image_size[0], image_size[1]

        boxes = outputs[0][:, :, :, 0:4]
        box_confidence = outputs[0][:, :, :, 4]
        box_class_probs = outputs[0][:, :, :, 5:]

        return boxes, box_confidence, box_class_probs
