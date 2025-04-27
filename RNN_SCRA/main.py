import math
import torch
import torch.nn as nn
from torio.utils.ffmpeg_utils import get_input_devices

from data_loader import data_loader as dl
import d2l

batch_size, num_steps = 32, 35
train_iter, vocab =dl(r"C:\Users\LENOVO\Desktop\code\AI\RNN\data\timemachine.txt",
                      'char',batch_size,num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size = shape, device = device)*0.01

    W_xh = normal((vocab_size,num_hiddens))
    W_hh = normal((num_hiddens,num_hiddens))
    b_h = torch.zeros(num_hiddens,device=device)
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    params = [W_xh,W_hh,b_h,W_hq,b_q]
    #params = [torch.tensor(x.clone().detach(), requires_grad = True,device = device) for x in param]
    for param in params:
        param.requires_grad_(True)
    return params
#print(get_params(num_steps,num_hiddens,'cuda:0'))
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X,W_xh) + torch.mm(H,W_hh) + b_h)
        Y = torch.mm(H,W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0),(H,)

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens,
                 device, get_params, initial_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.initial_state, self.forward_fn = initial_state, forward_fn
        self.params = get_params(vocab_size,num_hiddens,device)

    def __call__(self, X, state):
        X = nn.functional.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.initial_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
X = torch.arange(10).reshape((2, 5))
net = RNNModelScratch(len(vocab), num_hiddens, 'cuda:0', get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], 'cuda:0')
Y, new_state = net(X.to('cuda:0'), state)
#print(Y.shape, len(new_state), new_state[0].shape)

def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size = 1,device = device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]],device = device).reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.index_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm