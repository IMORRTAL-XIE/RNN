import d2l
import collections
import re

#torch.DATA_HUB['time_machine']=(torch.DATA_URL+'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()

if __name__ == '__main__':
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])


def tokenize(lines, token='char'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)

if __name__ == '__main__':
    for i in range(11):
        print(tokens[i])

#print(collections.Counter(['a','a','b','.','I_LOVE_DL !']))

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_token=None):
        if tokens is None :
            token = []
        if reserved_token is None :
            reserved_token = []
        counter = count_corpus(tokens)
        self._token_freq = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_token
        self.token_to_idx = { token:idx for idx,token in enumerate(self.idx_to_token)}
        for token,freq in self._token_freq :
            if freq <min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_token(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freq(self):
        return self._token_freq

def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
if __name__ == '__main__':
    print(vocab._token_freq[:10])

    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])