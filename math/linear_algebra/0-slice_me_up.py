#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 = [arr[0], arr[1]]  # your code here
len_arr = len(arr)
arr2 = arr[len_arr-4:len_arr] # your code here
arr3 = arr[1:5] # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
