import torch
import numpy as np

def post_processing(output1):
    threshold= 0.1
    notes= []
    this_onset= None
    this_offset= None
    for i in range(len(output1)):
        if output1[i][0][0] > threshold and this_onset == None:
            this_onset= i
        elif output1[i][0][1] > threshold and this_onset != None and this_onset+ 1 < i and this_offset == None:
            this_offset= i
            notes.append([this_onset* 0.032+ 0.016, this_offset* 0.032+ 0.016, 1])
            this_onset= None
            this_offset= None
    return notes