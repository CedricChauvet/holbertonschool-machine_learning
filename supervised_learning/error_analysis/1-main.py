#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels,logits)    
    
    print(sensitivity(confusion))
