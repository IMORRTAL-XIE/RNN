from corpus_loader import Vocabulary as Vocab
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

PATH = r"C:\Users\LENOVO\Desktop\code\AI\RNN\data\fra-eng\fra.txt"

def read_process_txt(path):
    with open(path, 'r', encoding = 'utf-8' ) as f:
        text = f.read()
        text = text.replace('\u202f',' ').replace('\xa0',' ').lower()
        def no_space(char, pre_char):
            return char in set(',.!?') and pre_char != ' '
        out = [' ' + char if i>0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
        return ''.join(out)

def tokenize(text, num_examples = None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def truncate_pad(line, num_steps, padding_token):
    if len(line)  > num_steps :
        return line[:num_steps]
    else :
        return line + [padding_token] * (num_steps - len(line))

def build_array(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.float32).sum(1)
    return array, valid_len

def data_loader(path, batch_size, num_steps, num_examples = 600):
    text = read_process_txt(path)
    source, target = tokenize(text, num_examples)
    src_vocab = Vocab(source, min_frequent=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_frequent=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    dataset = TensorDataset(*data_arrays)
    data_iter = DataLoader(dataset, batch_size, shuffle = True)
    return data_iter, src_vocab, tgt_vocab