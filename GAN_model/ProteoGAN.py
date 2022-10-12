import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils.torch_utils import *
from utils.spectral import SpectralNorm
from utils.gumbel import gumbel_softmax


class Generator(nn.Module):
    def __init__(self, hidden, z_dim=128, stride=8,
                 kernel_size=12, seq_len=512, n_chars=21):
        super(Generator, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(z_dim, hidden*n_chars),
            nn.ReLU(),
            nn.BatchNorm1d(hidden*n_chars)
        )
        L0, f0 = int(hidden/(stride**2)), int(n_chars*(stride**2))
        self.L, self.f = L0, f0
        f1 = f0//stride
        self.conv1 = nn.ConvTranspose1d(
            f0, f1, kernel_size, stride=stride, padding=2)
        self.conv2 = nn.ConvTranspose1d(
            f1, f1//stride, kernel_size, stride=stride, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.conv_block = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.BatchNorm1d(f1),
            self.conv2,
            nn.ReLU(),
            nn.BatchNorm1d(f1//stride),
        )
        # additional fc layer since MSA have non-fixed length
        self.fc1 = nn.Linear(hidden, seq_len)
        self.seq_len = seq_len

    def forward(self, x):
        B = x.size()[0]
        x = self.fc_block(x)
        x = x.view(-1, self.f, self.L)
        x = self.conv_block(x)  # B, C, H
        x = self.fc1(x)  # B, C, L
        x = torch.transpose(x, -1, -2)  # B, L, C
        x = x.contiguous()
        shape = x.size()
        x = x.view(B*self.seq_len, -1)
        x = gumbel_softmax(x, 0.5)
        return x.view(shape)


class Generator_BNbefore(nn.Module):
    # placing BN before ReLU, as in DCGAN
    def __init__(self, hidden, z_dim=128, stride=8,
                 kernel_size=12, seq_len=512, n_chars=21):
        super(Generator_BNbefore, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(z_dim, hidden*n_chars),
            nn.BatchNorm1d(hidden*n_chars),
            nn.ReLU()
        )
        L0, f0 = int(hidden/(stride**2)), int(n_chars*(stride**2))
        self.L, self.f = L0, f0
        f1 = f0//stride
        self.conv1 = nn.ConvTranspose1d(
            f0, f1, kernel_size, stride=stride, padding=2)
        self.conv2 = nn.ConvTranspose1d(
            f1, f1//stride, kernel_size, stride=stride, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.conv_block = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm1d(f1//stride),
            nn.ReLU(),
        )
        # additional fc layer since MSA have non-fixed length
        self.fc1 = nn.Linear(hidden, seq_len)
        self.seq_len = seq_len

    def forward(self, x):
        B = x.size()[0]
        x = self.fc_block(x)
        x = x.view(-1, self.f, self.L)
        x = self.fc1(self.conv_block(x))  # B, C, L
        x = torch.transpose(x, -1, -2)  # B, L, C
        x = x.contiguous()
        shape = x.size()
        x = x.view(B*self.seq_len, -1)
        x = gumbel_softmax(x, 0.5)
        return x.view(shape)


class Discriminator(nn.Module):
    def __init__(self, hidden, seq_len=512, n_chars=21):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(
            seq_len, hidden, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.conv2 = nn.Conv1d(hidden, 32, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.conv_block = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.2),
            SpectralNorm(self.conv2),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Linear(32*n_chars, 1)


    def forward(self, x):
        x = self.conv_block(x)
        #print(x.size())
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        return x
