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
        """
        Take the raw output : outputs
        return the bounding boxes ,Prob0 confidence and Prob class(80 len)
        """
        box_confidence = []
        boxes = []
        box_class_probs = []

        for i, output in enumerate(outputs):

            grid_height, grid_width, anchor_boxes, lastclasse = output.shape
            image_height, image_width = image_size[0], image_size[1]

            # found on internet
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))
            grid_x = grid_x.reshape(1, grid_height, grid_width, 1)
            grid_y = grid_y.reshape(1, grid_height, grid_width, 1)

            # center_x is the center, kept times image width
            center_x = (1 / (1 + np.exp(-output[..., 0])) + grid_x)\
                / grid_width * image_width
            # center_y is the center, kepts times image height
            center_y = (1 / (1 + np.exp(-output[:, :, :, 1])) + grid_y)\
                / grid_height * image_height
            # width of the bouding box beware the indexes 0 and 1
            width = self.anchors[i][:, 0] * np.exp(output[:, :, :, 2])\
                / self.model.input.shape[1] * image_width
            # pw is the weight
            height = self.anchors[i][:, 1] * np.exp(output[:, :, :, 3])\
                / self.model.input.shape[2] * image_height

            x1 = center_x - (width / 2)  # * image_width
            y1 = center_y - (height / 2)  # * image_height
            x2 = center_x + (width / 2)  # * image_width
            y2 = center_y + (height / 2)  # * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            # reshaping for the checker,
            confidence = 1 / (1 + np.exp(-output[:, :, :, 4]))
            confidence = confidence.reshape(grid_height, grid_width,
                                            anchor_boxes, 1)
            box_confidence.append(confidence)
            box_class_probs.append(1 / (1 + np.exp(-output[:, :, :, 5:])))

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        pass the output trough a filter,
        this threshold is  self.class_t * box_class_prob
        """

        threshold = self.class_t
        selected_BB = []
        selected_conf = []
        selected_Class = []

        # Filtrez le tableau
        for nb_output in range(len(box_confidences)):
            grid_height, grid_width, anchor_boxes, lastclasse = \
                box_confidences[nb_output].shape

            for i in range(grid_height):
                for j in range(grid_width):
                    for k in range(anchor_boxes):

                        index_C = box_class_probs[nb_output][i, j, k]\
                            .argmax()

                        max_class = box_class_probs[nb_output][i, j, k].max()

                        if box_confidences[nb_output][i, j, k, 0]\
                                * max_class > threshold:
                            selected_BB.append(boxes[nb_output][i, j, k, 0:4])
                            selected_Class.append(index_C)

                            # shapping of datas to fit box score :()
                            conf = box_confidences[nb_output][i, j, k]\
                                * box_class_probs[nb_output][i, j, k, index_C]
                            selected_conf.append(float(conf))

        selected_BB = np.array(selected_BB)
        selected_conf = np.array(selected_conf)
        selected_Class = np.array(selected_Class)
        return selected_BB, selected_Class, selected_conf
