import torch
import torch.nn as nn
import RNN
from grad_clipping import grad_clipping
import matplotlib.pyplot as plt

from data_loader import data_loader as ld
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

num_steps = 35
batch_size = 256
num_layer = 2
num_hiddens = 128
data_iter, vocab = ld(r"C:\Users\LENOVO\Desktop\code\AI\RNN\data\multiple.txt",'char', batch_size, num_steps, max_tokens = 10000)
input_size = output_size = len(vocab)
lstm_layer = nn.LSTM(input_size, num_hiddens, num_layer)
loss_fn = nn.CrossEntropyLoss()
losses = []
learning_rate = 1
epoch =10000

net = RNN.rnn(lstm_layer, input_size, num_hiddens, num_layer)
net = net.to('cuda:0')
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

def predict_ch8(prefix, num_preds, net, vocab):
    state = net.begin_state(1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device='cuda:0').reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int((torch.argmax(y,dim = 1)).reshape(1)))
    return ''.join([vocab.index_to_token[i] for i in outputs])

#print(predict_ch8('time', 10, net, vocab))


def train_epoch_ch8(net, data_iter, loss_fn, optimizer):
    state = None
    for X, Y in data_iter:
        if state is None:
            state = net.begin_state(X.shape[0])
        else:
            for s in state:
                s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to('cuda:0'), y.to('cuda:0')
        pred, state = net(X, state)
        loss = loss_fn(pred, y).mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(net, 1)
        optimizer.step()

for epoches in range(epoch):
    train_epoch_ch8(net, data_iter, loss_fn, optimizer)
    if (epoches+1) % 100 ==0 :
        print(f"已训练轮数：[{epoches+1}/{epoch}]")
        print(predict_ch8('the meaning of life is', 200, net, vocab))
        print(predict_ch8('life is', 200, net, vocab))

fig = plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(losses)),losses)
plt.show()

