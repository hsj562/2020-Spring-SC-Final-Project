import sys
import torch
import numpy as np

def testing(net, val_loader, pitch_set, device):
    net.eval()
    data = sample['data']
    data= torch.Tensor(data)

    target= sample['label']
    target= torch.Tensor(target)

    data= data.unsqueeze(1)
    target= target.unsqueeze(1)

    #print (data.shape)
    #print (target.shape)

    data_length= list(data.shape)[0]

    data = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)

    output1, output2 = net(data)
    #print (output1.shape)
    #print (output2.shape)
    answer= post_processing(output1, output2)

    net.eval()
    avg_score = []
    for batch_idx, sample in enumerate(val_loader):    
        raw_data  = sample['data']
        data= torch.FloatTensor(raw_data)
        #print(data.shappe)

        data = data.permute(1,0,2)
        data = data.cuda()

        # print (target.shape)

        #data_length= list(data.shape)[0]

        #target = target.to(device, dtype=torch.float)

        output1, output2 = net(data)
        #print(f'output1: {output1}, output2: {output2}')
        answer = post_processing(output1, sample['data'])
        return answer 
        answer = np.array(answer)
        target = target.numpy()

        ref_interval = answer[:, [0,1]]
        ref_pitch = answer[:, 2]
        est_interval = target[:, [0,1]]
        est_pitch = target[:, 2]

        score = evaluation(ref_interval, ref_pitch, est_interval, est_pitch )
       