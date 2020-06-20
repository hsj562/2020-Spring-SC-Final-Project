import torch.nn as nn
import torch
import time
import sys
import numpy as np
from eva import evaluate
from postprocess import post_processing

def do_training(net, loader, optimizer, device):
    num_epoch = 100
    criterion_onset= nn.BCELoss()
    #criterion_pitch= nn.L1Loss()
    train_loss= 0.0
    total_length= 0

    last = 0
    for epoch in range(num_epoch):
        net.train()
        total_length= 0.0
        print ("epoch %d start time: %f" %(epoch, time.time()))
        train_loss= 0.0
        score = 0
        for batch_idx, sample in enumerate(loader):
            data = sample['data']
            data= torch.Tensor(data)

            target= sample['label']
            target= torch.Tensor(target)

            data= data.permute(1,0,2)
            target= target.permute(1,0,2)

            data_length= list(data.shape)[0]

            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            output1 = net(data)

            res = post_processing(output1)

            total_loss = criterion_onset(output1, torch.narrow(target, dim= 2, start= 0, length= 2))
            train_loss = train_loss + total_loss.item()

            total_length= total_length+ 1
            total_loss.backward()
            optimizer.step()

            gt = post_processing(target)

            print(f'score: {evaluate(np.array(res), np.array(gt))}')
            
            if batch_idx % 50 == 0:
                print ("epoch %d, sample %d, loss %.6f" %(epoch, batch_idx, total_loss))
                sys.stdin.flush()

        print('epoch %d, avg loss: %.6f' %(epoch, train_loss/ total_length))
        
        if score > 0.25:
            model_path= f'ST_{epoch}.pt'
            torch.save(net.state_dict(), model_path)

    return net