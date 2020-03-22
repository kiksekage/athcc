import random


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


class VelocityNN(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, pad_length, layers):
        super(VelocityNN, self).__init__()
        if pad_length % 6 == 0:
            self.pad_length = pad_length
        else:
            raise Exception("Wrong size, needs to be divisible by 6")
        self.hidden_size = hidden_size
        self.layers = layers

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, self.layers, dropout=0.5)
        self.pool = nn.MaxPool1d(int(pad_length / 6))
        self.reducer = nn.Linear(int(pad_length / (pad_length / 6)), 1)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden=None, force=True, steps=0):
        inputs = self.input_layer(inputs.unsqueeze(1))
        output, hidden = self.rnn(inputs, hidden)
        output = self.pool(output.T).squeeze(1)
        output = F.leaky_relu(output)
        output = self.out(self.reducer(output).T)
        output = self.softmax(output.squeeze(1))
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(self.layers, 1, self.hidden_size)
        nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain("relu"))
        return hidden


def train(
    n_epochs, model, optimizer, criterion, train_data, pad_length, dimensions, save_file
):
    losses = []
    for epoch in range(n_epochs):
        print_loss_total = 0

        # Shuffle the data
        random.shuffle(train_data)

        # Go through each train point (should be batched...)
        for idx, train_point in enumerate(train_data):

            # Initialize hidden state and optimizer
            hidden_state = model.initHidden()
            optimizer.zero_grad()

            # Load in velocity data
            _inputs = train_point[0]
            inputs = Variable(torch.zeros(pad_length, dimensions).float())

            # Use up to pad_length of the velocity data
            if _inputs.shape[0] < pad_length:
                pad = _inputs.shape[0]
            else:
                pad = pad_length
            inputs[:pad, :] = torch.from_numpy(_inputs.values)[:pad, :]

            # Load in the label
            label = train_point[1]
            target = Variable(torch.from_numpy(np.asarray([label])))

            # Run the model
            outputs, hidden = model(inputs, hidden=hidden_state)

            # Take the class with highest confidence
            topv, topi = outputs.topk(1)
            guess = topi.squeeze().detach()

            # Loss calculation and backpropagation
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            print_loss_total += loss.item()

            if idx % 100 == 0:
                print_loss_avg = print_loss_total / 100
                losses.append(print_loss_avg)
                print(idx)
                print(print_loss_avg)
                print_loss_total = 0

    # Save model
    torch.save(model.state_dict(), save_file)


def evaluate(model, test_point, p_length, dim):
    with torch.no_grad():
        hidden_state = torch.zeros(model.layers, 1, model.hidden_size)

        _inputs = test_point[0]
        inputs = Variable(torch.zeros(p_length, dim).float())
        if _inputs.shape[0] < p_length:
            pad = _inputs.shape[0]
        else:
            pad = p_length
        inputs[:pad, :] = torch.from_numpy(_inputs.values)[:pad, :]
        label = test_point[1]
        target = Variable(torch.from_numpy(np.asarray([label])))

        outputs, hidden = model(inputs, hidden=hidden_state)

        topv, topi = outputs.topk(1)
        guess = topi.squeeze().detach()
        return guess
