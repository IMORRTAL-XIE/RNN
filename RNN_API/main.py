import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import predict_train as pt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
from data_loader import data_loader as ld

batch_size, num_steps = 32, 35
train_iter, vocab = ld("C:\\Users\\LENOVO\\Desktop\\code\\AI\\RNN\\data\\timemachine.txt",
                       'char', batch_size, num_steps)

num_hiddens = 256
num_layers = 2
rnn_layer = nn.LSTM(len(vocab), num_hiddens, num_layers)



class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = nn.functional.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

device = 'cuda:0'
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
pt.predict_ch8('time traveller', 10, net, vocab, device)

num_epochs, lr = 500, 1
pt.train_ch8(net, train_iter, vocab, lr, num_epochs, device)