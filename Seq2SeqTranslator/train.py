import torch
import torch.nn as nn
import math
import collections
from translator_data_loader import data_loader as ld
from translator_data_loader import truncate_pad as tp
import encoder_decoder as ed
from grad_clipping import grad_clipping as gc
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')


def sequence_mask(X, valid_len, value = 0):
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype = torch.float32, device = 'cuda:0').reshape(1, -1) < valid_len.reshape(-1, 1)
    X[~mask] = value
    return X

class MaskedCrossentropyLoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weighted_loss

def train_seq2seq(net, data_iter, learning_rate, epoch, tgt_vocab):
    """def xavier_init_weights(m):
        if type(m) == nn.Linear :
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU :
            for param in m._flat_weights_names :
                if "weight" in param :
                    nn.init.xavier_uniform_(m._parameters[param])
    """
    losses = []
    #net.apply(xavier_init_weights)
    net.to('cuda:0')
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    loss_fn = MaskedCrossentropyLoss()
    net.train()
    for epoches in range(epoch):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to('cuda:0') for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device = 'cuda:0').reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim = 1)
            pred, _ =net(X, dec_input, X_valid_len)
            loss = loss_fn(pred, Y, Y_valid_len)
            losses.append(loss.mean().item())
            loss.sum().backward()
            gc(net, 1)
            optimizer.step()
        if((epoches + 1) % 100 == 0):
            print(f'已训练轮次：[{epoches + 1}/{epoch}]')
    return losses

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights = False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device='cuda:0')
    src_tokens = tp(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype = torch.long, device = 'cuda:0'), dim = 0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype = torch.long, device = 'cuda:0'), dim = 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim = 2)
        pred = dec_X.squeeze(dim = 0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>'] :
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


PATH = r"C:\Users\LENOVO\Desktop\code\AI\RNN\data\fra-eng\fra.txt"
num_steps = 35
batch_size = 32
embed_size = 32
num_hiddens = 32
num_layers = 2
dropout = 0.1
learning_rate = 0.005
epoch = 500
data_iter, src_vocab, tgt_vocab = ld(PATH, batch_size, num_steps)
encoder = ed.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = ed.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = ed.EncoderDecoder(encoder, decoder)
losses =  train_seq2seq(net, data_iter, learning_rate, epoch, tgt_vocab)
fig = plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(losses)), losses)
plt.show()
print(losses[-1])
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

#torch.save(net, 'model.pth')