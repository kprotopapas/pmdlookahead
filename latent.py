import wandb
import torch
from torch.nn import *

from environments.solvable import Trap
from planning.vpn import VPN
from algorithms.policies import SoftmaxPolicy
from helper import parse_args

args = parse_args()

size = args.size
horizon = args.horizon
gamma = args.gamma or 1 - 1 / horizon
epochs = args.epochs
group = args.group

env = Trap(size, gamma)

pi = SoftmaxPolicy(env.observation_space, env.action_space)
V = env.V(pi)
Q = env.Q(pi)

Q_opt = env.optimal()
V_opt = Q_opt.max(axis=1)

n_z = env.n_obs
d_z = n_z // 4
depth = 5

model = VPN(n_z, d_z, depth=depth)

loss_z = MSELoss()
loss_r = MSELoss()
loss_v = MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

wandb.init(project='pmd-ts',
           config={'environment': {'name': env.name, 'gamma': env.gamma},
                   'algorithm': 'VPN'}, group=group)


for k in range(epochs):
    lz = torch.tensor(0.).double()
    lr = torch.tensor(0.).double()
    
    optimizer.zero_grad()

    states, actions, rewards = env.sample_trajectory(pi, horizon)
    rtgs = env.rtg(rewards)

    states, actions, rewards = torch.tensor(states), torch.tensor(actions), torch.tensor(rewards)
    rtgs = torch.tensor(rtgs)
    vk = rtgs.unfold(step=1, size=depth, dimension=-1)
    rk = rewards.unfold(step=1, size=depth, dimension=-1)

    rkh, vkh = model.forward(states, actions)

    lr = loss_r(rkh, rk)
    lv = loss_v(vkh, vk)

    loss = lr + lv
    
    loss.backward()

    optimizer.step()

    wandb.log({'lr': lr.item(), 'lv': lv.item(), 'samples': env.samples})