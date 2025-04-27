import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(10,32),nn.ReLU(),
            #nn.Linear(16,64),nn.ReLU(),
            #nn.Linear(64,16),nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        y=self.net(x)
        return y