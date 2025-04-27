import torch.nn as nn
import torch

class rnn(nn.Module):
    def __init__(self, rnn_layer, input_size, hidden_size, num_layer):
        super(rnn,self).__init__()
        self.rnn_layer = rnn_layer
        self.input_size = self.output_size =  input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.num_layer = num_layer

    def forward(self, x, state):
        X = nn.functional.one_hot(x.T.long(), self.input_size)
        X = X.to(torch.float32)
        Y, state = self.rnn_layer(X, state)
        output = self.linear(Y.reshape((-1,Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size):
        return (torch.zeros((self.num_layer, batch_size, self.hidden_size), device='cuda:0'),
                torch.zeros((self.num_layer, batch_size, self.hidden_size), device='cuda:0'))