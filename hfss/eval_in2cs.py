import sys
sys.path.insert(0,'../')

import torch
import numpy as np
from torchvision.transforms import transforms
import argparse
import pandas as pd
import os
os.environ["HF_ENDPOINT"] = 'http://hf-mirror.com' 
import pickle
import sys
from transforms_search_space import White_Mask

from utils import get_meanstd, get_dataset

repo_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0, repo_dir)

import timm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    stage = args.stage

    if args.dataset == 'sketch':
        sav_dir = 'sketch/'
        
        if not os.path.exists('sketch/'):
            os.makedirs('sketch/')            
        
    elif args.dataset == 'ImageNet_C':
        sav_dir = 'IN_C/'
        if not os.path.exists('IN_C/'):
            os.makedirs('IN_C/')
        
    elif args.dataset in ['imagenet_match', 'imagenet_thre', 'imagenet_top']  :
        sav_dir = 'INv2/'
        if  not os.path.exists('INv2/'):
            os.makedirs('INv2/')
        
    # if not os.path.exists(stage):
    #     os.makedirs(stage)
    print('loading shortcut masks')
    f = open(args.m_path+'.pkl', 'rb')
    DEFALUT_CANDIDATES = pickle.load(f)
    f.close()   

    for i in range(len(DEFALUT_CANDIDATES)):
        DEFALUT_CANDIDATES[i] = DEFALUT_CANDIDATES[i][0]

    batchsize = 32
    mean,std = get_meanstd()
    
    if args.model_path == 'cct_14_7x2_224':
        from src import cct_14_7x2_224
        model = cct_14_7x2_224(num_classes=1000,pretrained=True)
    else:
        model = timm.create_model(args.model_path,pretrained=True)
    model.to(device)
    model.eval()
    if  args.dataset == 'ImageNet_C':
        save_result = sav_dir + args.model_name_abbr + '_' + stage + args.trial_index +'severity'+args.severity  +args.corruption+ '.csv'
    else:
        save_result = sav_dir + args.model_name_abbr + '_' + stage + args.trial_index  +args.corruption+ '.csv'

    print(save_result)
    if os.path.exists(save_result):
        d = pd.read_csv(save_result, usecols=['Class_id', 'Org_acc','Org_DFM_acc','Corrupt_acc','Corrupt_DFM_acc'])
    else:
        kong =  [0 for i in range(1000)]
        d = {'Class_id': [i for i in range(1000)], 'Org_acc':kong,'Org_DFM_acc': kong, 
            'Corrupt_acc': kong, 'Corrupt_DFM_acc': kong
            }
        d = pd.DataFrame(data=d)
    s = 0 
    for s_i in range(1000):
        if d.loc[s_i,'Org_DFM_acc'] == 0 and d.loc[s_i,'Org_acc'] == 0 and d.loc[s_i,'Corrupt_DFM_acc'] == 0:
            # print('YYYY')
            s = s_i
            break
        else: 
            s = 1000


    for c_i in range(s,1000): 

        
        
        # print('class '+ str(c_i))
        chosen_mask = np.asarray(DEFALUT_CANDIDATES[c_i])
# -------------Original DFM

        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(args.img_size), transforms.ToTensor(),
                                      transforms.Normalize(mean, std),White_Mask(np.asarray(chosen_mask))])
        data_test = get_dataset(None,transform)
        indices = torch.tensor(data_test.targets)
        indices = (indices == c_i).nonzero() # if you want to keep images with the label 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        test_loader = torch.utils.data.DataLoader(data_test, batch_size= batchsize,sampler=sampler, shuffle=False,num_workers=4)
        

        num_correct = 0
        # print('Loaded data')
      
        for x,y in test_loader:
            x,y = x.to(device),y.to(device)    
            y_hat = model(x)
            pred = y_hat.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()

            
        d.loc[c_i,'Org_DFM_acc'] = (num_correct/len(indices))


        # Original 
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(args.img_size), transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
        data_test = get_dataset(None,transform)
        indices = torch.tensor(data_test.targets)
        indices = (indices == c_i).nonzero() # if you want to keep images with the label 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        test_loader = torch.utils.data.DataLoader(data_test, batch_size= batchsize,sampler=sampler, shuffle=False,num_workers=4)
        
  
        num_correct = 0
        for x,y in test_loader:
            x,y = x.to(device),y.to(device)    
            y_hat = model(x)
            
            pred = y_hat.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
        
        d.loc[c_i,'Org_acc'] = (num_correct/len(indices))



        transform=transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor(),
                                      transforms.Normalize(mean, std),White_Mask(np.asarray(chosen_mask))])
        data_test = get_dataset(args,transform)
        indices = torch.tensor(data_test.targets)
        indices = (indices == c_i).nonzero() # if you want to keep images with the label 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        test_loader = torch.utils.data.DataLoader(data_test, batch_size= batchsize,sampler=sampler, shuffle=False,num_workers=4)
        
        # print('Loaded data')
        num_correct = 0

        for x,y in test_loader:        
            x,y = x.to(device),y.to(device)    
            y_hat = model(x)
        
            pred = y_hat.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
        
        d.loc[c_i,'Corrupt_DFM_acc'] = (num_correct/len(indices))
    
        # print('Original corruption result')

        transform=transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
        data_test = get_dataset(args,transform)
        indices = torch.tensor(data_test.targets)
        indices = (indices == c_i).nonzero() # if you want to keep images with the label 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        test_loader = torch.utils.data.DataLoader(data_test, batch_size= batchsize,sampler=sampler, shuffle=False,num_workers=4)
        

        num_correct = 0
        # print('Loaded data')
      
        for x,y in test_loader:
            x,y = x.to(device),y.to(device)    
            y_hat = model(x)
            pred = y_hat.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
        
        d.loc[c_i,'Corrupt_acc'] = (num_correct/len(indices))
        #----------------------------------------

        
        d.to_csv(save_result)

   
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_path', type=str, default='./',
                        help='path of the msk')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset chosen')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--num_class', type=int, default=1000,
                        help='number of classes in dataset')
    parser.add_argument('--model_path', type=str, default='resnet18',
                        help='path of the model')
    parser.add_argument('--corruption', type=str, default='brightness',
                        help='corruption type in dataset')
    parser.add_argument('--stage', type=str, default='s6',
                        help='stage of DFM')
    parser.add_argument('--model_name_abbr', type=str, default='rn18',
                        help='model name')
    parser.add_argument('--trial_index', type=str, default='test1',
                        help='the first/second/.. trial of running HFSS')
    parser.add_argument('--severity', type=str, default='1',
                        help='severity of corruptions')
 
    args = parser.parse_args()

    main(args)
