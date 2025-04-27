from corpus_loader import load_corpus as lc
import torch
import random


def seq_data_iter_random(corpus,batch_size,num_steps):
    corpus = corpus[random.randint(0,num_steps-1):]
    num_subseqs = (len(corpus)-1) // num_steps
    initial_indices =list(range(0,num_subseqs*num_steps,num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos+num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0,num_batches*batch_size,batch_size):
        batch_initial_indices = initial_indices[i:i+batch_size]
        X = [data(j) for j in batch_initial_indices]
        Y = [data(j+1) for j in batch_initial_indices]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_sequential_in_batch(corpus,batch_size,num_steps):
    corpus = corpus[random.randint(0,num_steps-1):]
    num_subseqs = (len(corpus)-1) // num_steps
    initial_indices = list(range(0,num_steps*num_subseqs,num_steps))

    def data(pos):
        return corpus[pos:pos+num_steps]

    num_batches =  num_subseqs // batch_size
    for i in range(0,num_batches*batch_size,batch_size):
        batch_initial_indices = initial_indices[i:i+batch_size]
        X = [data(j) for j in batch_initial_indices]
        Y = [data(j+1) for j in batch_initial_indices]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_sequential_between_batch(corpus,batch_size,num_steps):
    corpus = corpus[random.randint(0,num_steps-1):]
    num_tokens = (len(corpus)-1) // batch_size * batch_size
    Xt = torch.tensor(corpus[:num_tokens])
    Yt = torch.tensor(corpus[1:num_tokens+1])
    Xt,Yt = Xt.reshape(batch_size,-1),Yt.reshape(batch_size,-1)
    num_batches = Xt.size(1) // num_steps
    for i in range(0,num_steps*num_batches,num_steps):
        X,Y = Xt[:,i:i+num_steps],Yt[:,i:i+num_steps]
        yield X,Y

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, path, token, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter == 0:
            self.data_iter_fn = seq_data_iter_random
        elif use_random_iter == 1:
            self.data_iter_fn = seq_data_iter_sequential_between_batch
        else :
            self.data_iter_fn = seq_data_iter_sequential_in_batch
        self.corpus, self.vocab = lc(path,token,max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def data_loader(path, token, batch_size, num_steps,
                           use_random_iter=0, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(path, token,
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

#c = list(range(35))
#for x,y in seq_data_iter_sequential_between_batch(c,2,5):
#    print(f"X:{x}\nY:{y}")
#for x,y in seq_data_iter_random(corpus,64,5):    print('X: ', x, '\nY:', y)