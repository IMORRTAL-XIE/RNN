import collections
import re

def read_txt(file:str) -> list[str] :
    """
    读取txt文件并且将非字母字符转化为空格，统一大小写。
    :param file: txt文件地址
    :return: 格式化文件
    """
    with open(file,'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

def tokenize(lines :list[str],token='word') -> list:
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char' :
        return [list(line) for line in lines]
    else :
        raise TypeError("未知词元类型！")

def corpus_counter(tokens):
    if len(tokens) == 0 :
        raise TypeError("请正确输入词元！")
    if isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocabulary:
    def __init__(self,tokens=None,min_frequent=0,reserved_tokens=None):
        if tokens is None :
            tokens = []
        if reserved_tokens is None :
            reserved_tokens = []
        counter =corpus_counter(tokens)
        self.token_frequent = sorted(counter.items(),key= lambda x:x[1],reverse=True)
        self.index_to_token = ['<unk>'] + reserved_tokens
        self.token_to_index = { token:index for index,token in enumerate(self.index_to_token)}
        for token,frequent in self.token_frequent:
            if frequent < min_frequent :
                break
            if token not in self.token_to_index:
                self.token_to_index[token] = len(self.index_to_token)
                self.index_to_token.append(token)

    def __len__(self):
        return len(self.index_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens,str):
            return [self.__getitem__(token) for token in tokens]
        else:
            return self.token_to_index.get(tokens,self.unk)

    def to_tokens(self,indices):
        if not isinstance(indices,int):
            return [self.to_tokens(index) for index in indices]
        else:
            return self.index_to_token[indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self.token_frequent

def load_corpus(path,token,max_token=-1):
    content = read_txt(path)
    tokens = tokenize(content,token)
    vocab = Vocabulary(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_token > 0 :
        corpus = corpus[:max_token]
    return corpus,vocab

"""
print(vocab.to_tokens([1,2,3,4,0,[1,2]]))
print(vocab['the',['is']])
print(tokens[:11])
ts=['a',[['b']]]
for l in ts:
    for t in l:
        print(t)
"""
"""
str = "adnakdn"
str = re.match("ad",str)
print(str)
"""