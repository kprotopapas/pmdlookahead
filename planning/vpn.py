import torch
from torch.nn import *

class VPN(torch.nn.Module):
    def __init__(self, n_z: int, d_z: int, depth: int):
        super().__init__()
        self.depth = depth
        hidden = min(max(16, d_z // 2), d_z)

        self.mu = Embedding(n_z, d_z) 

        
        self.latent = Sequential(Linear(d_z + 1, hidden, bias=False), ReLU(),
                                 Linear(hidden, hidden, bias=False), ReLU())
        
        self.latent2 = Sequential(Linear(d_z + 1, hidden, bias=False), ReLU(),
                                 Linear(hidden, hidden, bias=False), ReLU())
        
         
        self.gt = Sequential(self.latent,
                             Linear(hidden, d_z))
        
        
        self.rt = Sequential(self.latent,
                             Linear(hidden, 1))
        

        
        self.vt = Sequential(Linear(d_z, hidden, bias=False), ReLU(),
                             Linear(hidden, hidden, bias=False), ReLU(),
                             Linear(hidden, 1))

        self.double()

    
    def forward(self, states, actions):
        z0 = self.mu(states[:-1])[:-self.depth + 1]
        ak = actions.unfold(step=1, size=self.depth, dimension=-1)

        zk = []
        rk = []
        zi = z0
        for i in range(self.depth):
            ai = ak[:, i].reshape(-1, 1)
            hi = torch.hstack((zi, ai))

            zi = self.gt(hi)
            ri = self.rt(hi)
        
            zk.append(zi)
            rk.append(ri)
        
        zk = torch.dstack(zk).transpose(1, 2)
        rkh = torch.dstack(rk).transpose(1, 2)
        vkh = self.vt(zk)
        
        return rkh.squeeze(), vkh.squeeze()