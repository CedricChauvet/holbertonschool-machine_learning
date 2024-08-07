#!/usr/bin/env python3
"""
This is the yolo project
By Ced
"""
from keras.models import load_model
from keras.layers import Input
import keras
import numpy as np
import math
import cv2
import glob


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
        """
        version corrigées pour avancer sur le projet
        """

        keep_boxes = []

        for class_id in range(80):
            # Get indices of boxes for this class
            class_indices = np.where(box_classes == class_id)[0]

            if len(class_indices) == 0:
                continue

            # Sort boxes by score
            sorted_indices = class_indices[np.argsort(
                -box_scores[class_indices])]

            keep = []
            while len(sorted_indices) > 0:
                current = sorted_indices[0]
                keep.append(current)

                if len(sorted_indices) == 1:
                    break

                # Compute IoU of the picked box with the rest
                ious = np.array([IoU(filtered_boxes[current],
                                     filtered_boxes[i])
                                for i in sorted_indices[1:]])

                # Remove boxes with IoU over the threshold
                sorted_indices = sorted_indices[1:][ious < self.nms_t]

            keep_boxes.extend(keep)

        keep_boxes = np.array(keep_boxes)
        return filtered_boxes[keep_boxes], \
            box_classes[keep_boxes], box_scores[keep_boxes]

    def load_images(self, folder_path):
        """
        loads all images in a folder path
        """

        image_paths = glob.glob(folder_path + "/*.jpg")
        images = []

        for path in image_paths:
            images.append(cv2.imread(path, cv2.IMREAD_COLOR))

        return images, image_paths

    def preprocess_images(self, images):

        """
        cubic INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        create  2 np nd array pimage of dim=4
        and image_shape of dim 2
        """
        ni = len(images)
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages = np.ndarray([ni, input_h, input_w, 3])
        image_shapes = np.ndarray([ni, 2], dtype=int)
        for i, image in enumerate(images):
            image_shapes[i] = image.shape[0:2]

            resized_img = cv2.resize(
                image, (input_h, input_w), interpolation=cv2.INTER_CUBIC)
            rescaled_img = resized_img/255.0

            pimages[i] = rescaled_img

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        USING cv2 to display the boxes inside the image
        """
        n_boxes = boxes.shape[0]
        for i in range(n_boxes):

            start_point = (int(boxes[i, 0]), int(boxes[i, 1]))
            end_point = (int(boxes[i, 2]), int(boxes[i, 3]))
            color = (255, 0, 0)
            thickness = 2
            score = float("{0:.2f}".format(box_scores[i]))

            # passer du numéro d'etiquette au nom reel de l'objet
            with open('coco_classes.txt', 'r') as file:
                content = file.read().split("\n")
                box_name = content[box_classes[i]] + " " + str(score)
            # affiche le rectangle de la bounding box
            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=color,
                                  thickness=thickness)

            # ajoute le nombe de l objet et la confidence
            image = cv2.putText(image, box_name,
                                org=(int(boxes[i, 0]), int(boxes[i, 1] - 5)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 0, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA)

        cv2.imshow(file_name, image)

        while True:
            # Wait for the S key, if s is pressed save the pitcure
            if cv2.waitKey(0) & 0xFF == 115:
                cv2.imwrite(file_name, image)

                break
            if cv2.waitKey(0) & 0xFF != 115:
                break
        cv2.destroyAllWindows()  # destroys the window showing image

        return

    def predict(self, folder_path):
        """
        c'est la tache qui me pose des difficultés,
        prendre plusieurs images et dessiner les boundings boxes
        """

        # charge les images dans le dossier
        images, image_paths = self.load_images(folder_path)
        save = []

        for file_name in image_paths:
            file_name = file_name.split("/")
            save.append(file_name[-1])

        pimages, image_shapes = self.preprocess_images(images)
        for i, img in enumerate(pimages):

            # mon probleme se situe ici!
            output = self.model(np.expand_dims(img, axis=0))

            output1 = output[0][0, :, :, :]
            output2 = output[1][0, :, :, :]
            output3 = output[2][0, :, :, :]
            image_size = (img.shape[0], img.shape[1])

            boxes, box_confidences, box_class_probs = self.process_outputs(
                [output1, output2, output3], image_size)

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)

            boxes, box_classes, box_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)
            self.show_boxes(img, boxes, box_classes, box_scores, save[i])


def IoU(BB1, BB2):
    """
    get iou factor
    """
    x0_A, y0_A, x1_A, y1_A = BB1
    x0_B, y0_B, x1_B, y1_B = BB2

    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    # Calculate width and height of the intersection area.
    width_I = x1_I - x0_I
    height_I = y1_I - y0_I
    # Handle the negative value width or height of the intersection area
    if width_I < 0:
        width_I = 0
    if height_I < 0:
        height_I = 0

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
