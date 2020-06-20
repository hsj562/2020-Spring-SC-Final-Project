import json
import os
import numpy as np
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

from training import do_training
from preprocess import preprocess
#os.environ["CUDA_VISIBLE_DEVICES"]="3" 

class Net(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=5, dropout=0.01):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.Linear1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, 1)

    def forward(self, input_data):
        out = F.relu(self.Linear1(input_data))
        out, _ = self.rnn(out)
        out = torch.sigmoid(self.Linear2(out))
        return out


class MyData(Data.Dataset):
    def __init__(self, data_seq, label):
        self.data_seq = data_seq
        self.label= label

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
            'label': self.label[idx]
        }

def collate_fn(samples):
    batch = {}
    #print (samples[0]['data'].shape)
    temp= [torch.from_numpy(np.array(sample['data'], dtype= np.float32)) for sample in samples]
    padded_data = rnn_utils.pad_sequence(temp, batch_first=True, padding_value= 0)
    batch['data']= padded_data
    batch['label']= [np.array(sample['label'], dtype= np.float32) for sample in samples]
    return batch

feat = ['zcr', 'energy', 'spectral_centroid', 'spectral_flux', 'chroma_2', 'chroma_4', 'chroma_6', 'chroma_8', 'chroma_11', 'chroma_12', 'vocal_pitch', 'time']

TEST = "./AIcup_testset_ok"
TRAIN = "./MIR-ST500"

if __name__ == '__main__':
    THE_FOLDER = TRAIN

    for the_dir in os.listdir(THE_FOLDER):
        data_seq = []
        label = []
        print (the_dir)
        if not os.path.isdir(THE_FOLDER + "/" + the_dir):
            continue

        json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"
        
        # for train
        gt_path= THE_FOLDER+ "/" +the_dir+ "/"+ the_dir+ "_groundtruth.txt"
        
        youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

        with open(json_path, 'r') as json_file:
            temp = json.loads(json_file.read())
        # for train
        gtdata = np.loadtxt(gt_path)
        
        data = []
        for key, value in temp.items():
            if key in feat:
                data.append(value)

        data= np.array(data).T
        data_seq.append(data)
        # for train
        label.append(gtdata)
     
    label = preprocess(data_seq, label)

    train_data = MyData(data_seq, label) # modify when test    
    with open('feature_pickle.pkl', 'wb') as pkl:
        pickle.dump(train_data, pkl)

    train_data = None
    with open('feature_pickle.pkl', 'rb') as pkl:
        train_data = pickle.load(pkl)

    input_dim= len(feat)
    hidden_size= 50

    BATCH_SIZE= 1
    loader = Data.DataLoader(dataset=train_data, batch_size= BATCH_SIZE, collate_fn = collate_fn, shuffle=True, num_workers = 6)

    model = Net(input_dim, hidden_size)#.cuda()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    device = 'cpu'

    # if torch.cuda.is_available():
    #     device = 'cuda'
    # print('use', device, 'now')

    model.to(device)
    model = do_training(model, loader, optimizer, device)
    
    #for testing
    '''
    model.load_state_dict(torch.load("ST_10.pt"))
    ans = testing(model, loader, pitch_set, 'cuda')
    json.dump(ans, codecs.open(f'./json/output{idx}.json', 'w', encoding='utf-8'), separators=(',',':'), sort_keys=True, indent=4)
    idx+=1
    '''
