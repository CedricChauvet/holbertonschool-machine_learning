#!/usr/bin/env python3
"""
task 4, project plotting: Frequency
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ This is a documentation"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.hist(student_grades.astype("int32"),bins = [10,20,30,40,50,60,70,80,90,100], edgecolor="black")
    plt.xlim(0,100)
    plt.xticks(np.arange(0,101,10))
    plt.yticks(np.arange(0,31,5))
    plt.show()
