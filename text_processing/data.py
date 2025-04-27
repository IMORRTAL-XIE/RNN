import torch
from matplotlib.pyplot import xscale, yscale

import d2l
from main import read_time_machine as r
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
import random

tokens = d2l.tokenize(r())

corpus = list(token for line in tokens for token in line)

vocab = d2l.Vocab(corpus)
bio_token = [pair for pair in zip(corpus[:-1],corpus[1:])]
tri_token = [pair for pair in zip(corpus[:-2],corpus[1:-1],corpus[2:])]
print(bio_token[:11])
print(tri_token[:11])
bio_vocab = d2l.Vocab(bio_token)
tri_vocab = d2l.Vocab(tri_token)
print(vocab.token_freqs[:11])
print(bio_vocab.token_freqs[:11])
print(tri_vocab.token_freqs[:11])
fig1 = plt.figure()
freq = [freq for token,freq in vocab.token_freqs]
bio_freq = [b for token,b in bio_vocab.token_freqs]
tri_freq = [t for token,t in tri_vocab.token_freqs]
plt.plot(torch.arange(len(freq)),freq,label='unigram',)
plt.plot(torch.arange(len(bio_freq)),bio_freq,label='bigram')
plt.plot(torch.arange(len(tri_freq)),tri_freq,label='trigram')
plt.xscale('log')
plt.yscale('log')
plt.show()