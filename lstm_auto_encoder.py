from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.DataLoader

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence,self).__init__()
        self.lstm_enc = nn.LSTMCell(1,hidden_size=128)
        self.fc_enc = nn.Linear(128,1)
        self.lstm_dec = nn.LSTMCell(1,128)
        self.fc_dec = nn.Linear(128,1)

    def forward(self,input,input_reverse):
        outputs = []
        o_t_enc = Variable(torch.zeros(input.size(0),128).cuda(), requires_grad=False)
        h_t_enc = Variable(torch.zeros(input.size(0),128).cuda(), requires_grad=False)

        for i,input_t in enumerate(input.chunk(input.size(1),dim=1)):
            o_t_enc,h_t_enc = self.lstm_enc(input_t,(o_t_enc,h_t_enc))
            output = self.fc_enc(h_t_enc)

        outputs += [output]

        for i, input_t in enumerate(input_reverse.chunk(input_reverse.size(1),dim=1)):
            for i in range( input_reverse.size(1)-1):
                o_t_dec,h_t_dec = self.lstm_dec(input_t,(o_t_enc,h_t_enc))
                output = self.fc_dec(h_t_dec)
                outputs += [output]

        outputs = torch.stack(outputs,1).squeeze(2)

        return outputs

model = Sequence()

batch_size=50

data_file =open('/home/sammer/Master_Thesis/Research/Code/videogan/data/list.txt','r')
dataset = []
for line in data_file:
    dataset.append(line)

class ImageDataSet(torch.utils.Dataset):
    def __init__(self):
        self.data_files = dataset
        torch.sort(self.data_files)

    def __getindex__(self, idx):
        return self.data_files[idx]

    def __len__(self):
        return len(self.data_files)

dset =ImageDataSet()
loader = torch.utils.DataLoader(dset,num_workers=8)



