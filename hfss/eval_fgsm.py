import torch
import torch.nn.functional as F
from torchvision import  transforms
import numpy as np
import os
import timm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.datasets import ImageFolder
import pickle
import pandas as pd
from transforms_search_space import White_Mask
import argparse
import sys
from utils import get_meanstd, get_dataset, denorm, fgsm_attack

repo_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0, repo_dir)

def main(args):


    if not os.path.exists('FGSM/'):
        os.makedirs('FGSM/')
    sav_dir = 'FGSM/'

    
    print('loading shortcut masks')
    f = open(args.m_path+'.pkl', 'rb')
    DEFALUT_CANDIDATES = pickle.load(f)
    f.close()   

    for i in range(len(DEFALUT_CANDIDATES)):
        DEFALUT_CANDIDATES[i] = DEFALUT_CANDIDATES[i][0]

    stage = args.stage
    batchsize = 1
    mean,std = get_meanstd()
    
    if args.model_path == 'cct_14_7x2_224':
        from src import cct_14_7x2_224
        model = cct_14_7x2_224(num_classes=1000,pretrained=True)
    else:
        model = timm.create_model(args.model_path,pretrained=True)
    model.to(device)
    model.eval()
    torch.manual_seed(42)

    save_result = sav_dir + args.model_name_abbr + '_' + stage + args.trial_index  +args.corruption+ '.csv'
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
# -------------Atack DFM

        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
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
            x.requires_grad = True
            y_hat = model(x)

            init_pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if init_pred.item() != y.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(y_hat, y)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect ``datagrad``
            data_grad = x.grad.data

            # Restore the data to its original scale
            data_denorm = denorm(x)

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data_denorm, 8/255.0, data_grad)

            # Reapply normalization
            perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

            # Re-classify the perturbed image
            output = model(perturbed_data_normalized)
            pred = output.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
        
        # print(torch.sum(Matrix2))
        
        d.loc[c_i,'Corrupt_DFM_acc'] = (num_correct/len(indices))

#-----------------------------------------------Attack on original ------
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
        data_test = get_dataset(None,transform)
        indices = torch.tensor(data_test.targets)
        indices = (indices == c_i).nonzero() # if you want to keep images with the label 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        test_loader = torch.utils.data.DataLoader(data_test, batch_size= batchsize,sampler=sampler, shuffle=False,num_workers=4)
        

        num_correct = 0 
        # print('Loaded data')
      
        for x,y in test_loader:
            x,y = x.to(device),y.to(device)    
            x.requires_grad = True
            y_hat = model(x)

            init_pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if init_pred.item() != y.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(y_hat, y)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect ``datagrad``
            data_grad = x.grad.data

            # Restore the data to its original scale
            data_denorm = denorm(x)

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data_denorm, 8/255.0, data_grad)

            # Reapply normalization
            perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

            # Re-classify the perturbed image
            output = model(perturbed_data_normalized)
            pred = output.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
        
        # print(torch.sum(Matrix2))
        
        d.loc[c_i,'Corrupt_acc'] = (num_correct/len(indices))
    #-----------------------Original DFM-------------

        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean, std),White_Mask(np.asarray(chosen_mask))])
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
            
        # print(torch.sum(Matrix2))
        d.loc[c_i,'Org_DFM_acc'] =(num_correct/len(indices))


        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
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
            
        # print(torch.sum(Matrix2))
        d.loc[c_i,'Org_acc'] = (num_correct/len(indices))

        d.to_csv(save_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_path', type=str, default='./',
                        help='path of the msk')
    parser.add_argument('--dataset', type=str, default='fgsm',
                        help='dataset chosen')
    parser.add_argument('--model_path', type=str, default='resnet18',
                        help='path of the model')
    parser.add_argument('--corruption', type=str, default='fgsm',
                        help='corruption type in dataset')
    parser.add_argument('--stage', type=str, default='s6',
                        help='stage of DFM')
    parser.add_argument('--model_name_abbr', type=str, default='rn18',
                        help='model name')
    parser.add_argument('--trial_index', type=str, default='test1',
                        help='the first/second/.. trial of running HFSS')


    args = parser.parse_args()

    main(args)






