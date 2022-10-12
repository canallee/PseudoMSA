import torch
import numpy as np
import torch.nn as nn
from utils.torch_utils import *
from utils.spectral import SpectralNorm
from utils.gumbel import gumbel_softmax


class ResBlockG(nn.Module):
    def __init__(self, hidden):
        # input: BxDxL
        super(ResBlockG, self).__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.model = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            self.conv1,
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            self.conv2,
        )
        self.bypass = nn.Sequential()

    def forward(self, x):
        # in: B,D,L, out: B,D,L
        return self.model(x) + self.bypass(x)


class ResBlockD(nn.Module):
    def __init__(self, hidden, stride=1):
        super(ResBlockD, self).__init__()

        self.conv1 = nn.Conv1d(hidden, hidden, 3,  padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.conv2 = nn.Conv1d(hidden, hidden, 3,  padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.1),
                SpectralNorm(self.conv1),
                nn.LeakyReLU(0.1),
                SpectralNorm(self.conv2)
            )
            self.bypass = nn.Sequential()
        else:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.1),
                SpectralNorm(self.conv1),
                nn.LeakyReLU(0.1),
                SpectralNorm(self.conv2),
                nn.AvgPool1d(2, stride=stride, padding=0)
            )

            self.bypass_conv = nn.Conv1d(hidden, hidden, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool1d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockD_0(nn.Module):
    def __init__(self, hidden, n_chars):
        super(ResBlockD_0, self).__init__()
        self.conv1 = nn.Conv1d(n_chars, hidden, 3,  padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.conv2 = nn.Conv1d(hidden, hidden, 3,  padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.bypass_conv = nn.Conv1d(n_chars, hidden, 1, padding=0)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.1),
            SpectralNorm(self.conv2),
            nn.AvgPool1d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool1d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResNetSN_Generator(nn.Module):
    def __init__(self, hidden, seq_len=512, n_chars=21, z_dim=128):
        super(ResNetSN_Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlockG(hidden),
            ResBlockG(hidden),
            ResBlockG(hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, n_chars, 1)
        )
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden

    def forward(self, noise):
        B = noise.size()[0]
        output = self.fc1(noise)  # (B, D, L)
        output = output.view(-1, self.hidden, self.seq_len)
        output = self.block(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(B*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape)  # (B, L, C)


class ResNetSN_Discriminator(nn.Module):
    def __init__(self, hidden, seq_len=512, n_chars=21):
        super(ResNetSN_Discriminator, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlockD_0(hidden, n_chars),  # downsampling by 2
            ResBlockD(hidden, stride=2),
            ResBlockD(hidden, stride=2),
            ResBlockD(hidden, stride=2),
            nn.LeakyReLU(0.1),
            # sequence length may not be multiple of 2
            nn.AvgPool1d(seq_len//2//2//2//2)
        )
        self.fc = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, input):
        output = input.transpose(1, 2)  # BxLxC -> BxCxL
        output = self.block(output)
        output = output.view(-1, self.hidden)
        output = self.fc(output)
        return output

# class ResNetSN_Generator(nn.Module):
#     def __init__(self, batch_size, hidden, seq_len=512, n_chars=21, z_dim=128, debug=False):
#         super(ResNetSN_Generator, self).__init__()
#         self.fc1 = nn.Linear(z_dim, hidden*seq_len)

#         self.block = nn.Sequential(
#             ResBlockG(hidden),
#             ResBlockG(hidden),
#             ResBlockG(hidden),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(),
#             nn.Conv1d(hidden, n_chars, 1)
#         )

#         self.n_chars = n_chars
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.hidden = hidden
#         self.debug = debug

#     def forward(self, noise):
#         if self.debug:
#             step = 0
#             print('step',step, noise.size()); step+=1
#             output = self.fc1(noise)
#             # (B, D, L)
#             print('step',step, output.size()); step+=1
#             output = output.view(-1, self.hidden, self.seq_len)
#             print('step',step, output.size()); step+=1
#             output = self.block(output)
#             print('step',step, output.size()); step+=1
#             output = output.transpose(1, 2)
#             print('step',step, output.size()); step+=1
#             shape = output.size()
#             print('step',step, output.size()); step+=1
#             output = output.contiguous()
#             print('step',step, output.size()); step+=1
#             output = output.view(self.batch_size*self.seq_len, -1)
#             print('step',step, output.size()); step+=1
#             output = gumbel_softmax(output, 0.5)
#             print('step',step, output.size()); step+=1
#         else:
#             output = self.fc1(noise)
#             # (B, D, L)
#             output = output.view(-1, self.hidden, self.seq_len)
#             output = self.block(output)
#             output = output.transpose(1, 2)
#             shape = output.size()
#             output = output.contiguous()
#             output = output.view(self.batch_size*self.seq_len, -1)
#             output = gumbel_softmax(output, 0.5)
#         return output.view(shape)  # (B, L, C)

# class ResNetSN_Discriminator(nn.Module):
#     def __init__(self, batch_size, hidden, seq_len=512, n_chars=21, debug=False):
#         super(ResNetSN_Discriminator, self).__init__()
#         self.n_chars = n_chars
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.hidden = hidden
#         self.block = nn.Sequential(
#             ResBlockD_0(hidden, n_chars), # downsampling by 2
#             ResBlockD(hidden, stride=2),
#             ResBlockD(hidden, stride=2),
#             ResBlockD(hidden, stride=2),
#             nn.LeakyReLU(0.1),
#             # sequence length may not be multiple of 2
#             nn.AvgPool1d(seq_len//2//2//2//2)
#         )
#         self.debug = debug
#         self.fc = nn.Linear(hidden, 1)
#         nn.init.xavier_uniform_(self.fc.weight.data, 1.)
#         self.fc = SpectralNorm(self.fc)

#     def forward(self, input):
#         if self.debug:
#             step = 0
#             print('step',step, input.size()); step+=1
#             output = input.transpose(1, 2)  # BxLxC -> BxCxL
#             print('step',step, output.size()); step+=1
#             output = self.block(output)
#             print('step',step, output.size()); step+=1
#             output = output.view(-1, self.hidden)
#             print('step',step, output.size()); step+=1
#             output = self.fc(output)
#             print('step',step, output.size()); step+=1
#         else:
#             output = input.transpose(1, 2)  # BxLxC -> BxCxL
#             output = self.block(output)
#             output = output.view(-1, self.hidden)
#             output = self.fc(output)
#         return output
