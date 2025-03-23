import os
import fire
import time
import random
from pprint import pprint
import torch
import torch.backends.cudnn as cudnn
from utils import parse_args
from utils import find_SI
import sys
repo_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0, repo_dir)
 


import timm
 
def train(**kwargs):
    print('\n[+] Parse arguments')
    args, kwargs = parse_args(kwargs)
    pprint(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    print('\n[+] Create network')
    # Customize her your model
    model = timm.create_model(args.model_path,pretrained=True)

   
    if args.use_cuda:
        model = model.cuda()

    print('\n[+] Load dataset')

    start = time.time()
    find_SI(args, model.eval())    
    end = time.time()
    print('%.3f seconds' %(end - start))

if __name__ == '__main__':
    
    fire.Fire(train)
    
