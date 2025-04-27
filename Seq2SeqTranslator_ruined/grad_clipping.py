import torch
def grad_clipping(net, theta):
    params = [param for param in net.parameters() if param.requires_grad]
    norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in params))
    if norm > theta :
        for param in params:
            param.grad[:] *= theta/norm
    return