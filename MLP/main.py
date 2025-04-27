import torch
import torch.utils.data.dataloader
import torch.nn as nn
from sympy.printing.pretty.pretty_symbology import line_width

import MLP
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

data_size=2000
tau=10
time=torch.arange(0,data_size,dtype=torch.float32)
X=torch.sin(time*0.01)+torch.normal(0,0.2,[data_size])

"""fig1=plt.figure(figsize=(6,3))
plt.plot(time,X)
plt.xlabel('time')
plt.ylabel('X')
plt.show()"""

x=torch.zeros([data_size-tau,tau]).to('cuda:0')
for i in range(tau):
    x[:,i]=X[i:data_size-tau+i]
#print(x)

x.to('cuda:0')
y=X[tau:data_size].reshape(-1,1).to('cuda:0')
#print(y.size(0),y.size(1))

train_x=x[:int((data_size-tau)*0.7),:]
train_y=y[:int((data_size-tau)*0.7),:]
test_x=x[int((data_size-tau)*0.7):,:]
test_y=y[int((data_size-tau)*0.7):,:]

#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

model=MLP.MLP().to('cuda:0')
epochs=3000
learning_rate=0.01
loss_fn=nn.MSELoss()
losses=[]
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    Pred=model(train_x)
    loss=loss_fn(Pred,train_y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
Pred=Pred.cpu()
train_x=train_x.cpu()
fig0=plt.figure()
plt.plot(range(0,int((data_size-tau)*0.7)),Pred.detach().numpy())
plt.plot(range(0,int((data_size-tau)*0.7)),train_x[:,0],linewidth=0.1)
plt.show()
fig2=plt.figure()
plt.plot(range(epochs), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



with torch.no_grad():
    Pred=model(test_x)
    loss=loss_fn(Pred,test_y)
    print(loss.item())
Pred=Pred.cpu()
test_y=test_y.cpu()
fig3=plt.figure()
plt.plot(range(int((data_size-tau)*0.7)+5+6,data_size+1),test_y)
plt.plot(range(int((data_size-tau)*0.7)+6+5,data_size+1),list(Pred.numpy()))
plt.xlabel('time')
plt.ylabel('prediction')
plt.show()

mul_pre=torch.zeros(data_size)
mul_pre[:int((data_size-tau)*0.7)+tau]=X[:int((data_size-tau)*0.7)+tau]
mul_pre.to('cuda:0')
with torch.no_grad():
    for i in range(int((data_size-tau)*0.7)+tau,data_size):
        mul_pre[i]=model(mul_pre[i-tau:i].reshape(1,-1).to('cuda:0'))
fig5=plt.figure()
mul_pre=mul_pre.cpu()
plt.plot(range(int((data_size-tau)*0.7)+tau,data_size),mul_pre[int((data_size-tau)*0.7)+tau:].numpy())
plt.plot(range(int((data_size-tau)*0.7)+tau,data_size),X[int((data_size-tau)*0.7)+tau:].numpy())
plt.show()