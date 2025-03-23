import numpy as np
import torch
import random
import torch.fft as fft


def filter(x,mask):
    x1 = x 
    y1 = torch.zeros(x1.size(),dtype=torch.complex128)
    y1 = fft.fftshift(fft.fft2(x1)) 
    for j in range(3):
        y1[j,:,:] = y1[j,:,:]* mask
    x1_w = fft.ifft2(fft.ifftshift(y1))
    return torch.Tensor(torch.real(x1_w))


class White_Mask(object):
    def __init__(self, mask, flip = False):
       
        self.mask = mask
        self.flip = flip
             
        
    def __call__(self, x):
       
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        mask = self.mask
        if self.flip:
            mask = 1-mask
        x = filter(x,mask)
        
        return x.type(torch.float32) 
    def mask(self):
        return self.mask



def gen_freqs_list(h,w):
    fs = []
    if h %2 == 0:
        limit = int(np.floor(h / 2.0))
    else:
        limit = int(np.floor(h / 2.0))+1
    for h_index in range(limit):    
            for w_index in range(w):
                fs.append([h_index,w_index])
    return fs

def sample_frequency(portion, f_list):
    tot = len(f_list)-1
    res = set()
    N=int(len(f_list)*portion)
    for _ in range(N):
        temp = random.randint(0, tot) 
        while any(temp == idx  for idx in res):
            temp = random.randint(0, tot) 
        res.add(temp)
    res = [idx for idx in res]
    
    select_freqs = [f_list[i]  for i in res] 
    return  select_freqs


def generate_mask(sel_freqs, h,w):
    m = torch.zeros((h,w))
    for i in sel_freqs:
        m[i[0],i[1]] = 1
    return m
