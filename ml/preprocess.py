import numpy as np
import torch

def preprocess(data_seq, label):
    new_label= []
    for i in range(len(label)):
        label_of_one_song= []
        cur_note= 0
        cur_note_onset= label[i][cur_note][0]
        cur_note_offset= label[i][cur_note][1]
        cur_note_pitch= label[i][cur_note][2]

        for j in range(len(data_seq[i])):
            cur_time= j* 0.032+ 0.016
        
            if abs(cur_time - cur_note_onset) < 0.017:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))

            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0, 0.0]))

            elif abs(cur_time - cur_note_offset) < 0.017:
                label_of_one_song.append(np.array([0, 1, cur_note_pitch]))
                cur_note= cur_note+ 1
                if cur_note < len(label[i]):
                    cur_note_onset= label[i][cur_note][0]
                    cur_note_offset= label[i][cur_note][1]
                    cur_note_pitch= label[i][cur_note][2]
            else:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))

        new_label.append(label_of_one_song)

    return new_label
