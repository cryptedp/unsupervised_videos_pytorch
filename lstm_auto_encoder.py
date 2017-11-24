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
import torch.utils.data

import torch.utils.DataLoader


#LSTM autoencoder model
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


##load data
path = "."
activity_net_data = datasets.ImageFolder(path)
data_loader = torch.utils.data.DataLoader(activity_net_data,batch_size=64,shuffle=True,num_workers=64)


##train model


for p in model.parameters():
    print('parameters ', p.size())

# optimizers = ["op1","op2","op3"]
# foreach op in ops (so is all in one place)
# optimizer = optim.Adadelta(model.parameters(), lr=0.01)
optimizers = {'SGD': optim.SGD(model.parameters(), lr=0.01)}

for name, optimizer in optimizers.items():
    model.train()
    train_loss = []
    train_accu = []
    i = 0
    for epoch in range(10):
        for data in data_loader :
              # use cuda if available
            img,_ = data
            optimizer.zero_grad()
            output = model.forward(data)

            img = Variable(img).cuda()
            loss = F.mse_loss(img,output)
            loss.backward()  # calc gradients
            train_loss.append(loss.data[0])
            optimizer.step()  # update gradients
            prediction = output.data.max(1)[1]  # first column has actual prob.

            print('epoch [{}/{}], loss:{:.4f}'
                .format(epoch + 1, 10, loss.data[0]))

torch.save(model.state_dict(), './lstm_autoencoder.pth')
