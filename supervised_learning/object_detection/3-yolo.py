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


    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        # we start with class zero, wich is the minimum, to 79    
        tuple_de_sortie=np.array([],dtype=int)
        for number_class in range(80):
            classified_index=[]
            classified_index_nms=[]
            #min_class contient tout les element d'un casse spécifique
            min_class =  np.where(box_classes == number_class)
            #pour faciliter l'usage
            class_rest = min_class[0]

            while( len(class_rest)>=1):

                # find the best score in min_class
                index_max_score = box_scores[class_rest].argmax()
                best_index = class_rest[index_max_score]
                class_rest = np.delete(class_rest,index_max_score)
                classified_index.append(best_index)               
                

            # création d'un tampon pour recuperer les nms
            classified_index_nms=classified_index
            
            #compare le premier element i0 de la liste qui a le meilleur
            # score au reste de la liste
            for len_class in range(len(classified_index)):
                i_0 = classified_index[len_class] 
                #print("i0 score", box_scores[i_0], "classe", box_classes[i_0])
                for index_i, i in enumerate(classified_index[len_class+1:]):
                    # print("i score   > ", box_scores[i], "classe",box_classes[i])
                    if IoU(filtered_boxes[i_0],filtered_boxes[i]) > self.nms_t:
                        classified_index_nms = np.delete(classified_index_nms,index_i)
                       
            tuple_de_sortie = np.append(tuple_de_sortie, classified_index_nms, axis = None)
        # classé de la + grande boxe score a la plus peitite sur une classe precise
            print("tuple", tuple_de_sortie)
        return filtered_boxes[tuple_de_sortie], box_classes[tuple_de_sortie], box_scores[tuple_de_sortie]
    
    
    
    
    
    
    
def IoU(BB1, BB2):
    x0_A, y0_A, x1_A, y1_A = BB1
    x0_B, y0_B, x1_B, y1_B = BB2
    
        # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    #Calculate width and height of the intersection area.
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I
    # Handle the negative value width or height of the intersection area
    if width_I<0 : width_I=0
    if height_I<0 : height_I=0
    
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection/union

    # Return the IoU and intersection box
    return IoU