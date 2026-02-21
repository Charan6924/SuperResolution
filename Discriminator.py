import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
    )

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
    def forward(self,x):
        return x
    