 
import json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
import pickle

from transforms_search_space import White_Mask,gen_freqs_list,sample_frequency,generate_mask
from utils import split_dataset, get_dataset,get_dataloader
from utils import parse_args,validate_classwise
 

def get_candidates(args,num_candidates,portion,stage):
 
    DEFALUT_CANDIDATES = []

    max_n_h = int(np.floor(args.img_size / 2.0))
    max_n_w = int(np.floor(args.img_size / 2.0))
    if args.img_size == 224:
        patches = {'stage1': 4, 'stage2': 4*2, 'stage3': 4*2*2, 'stage4': 28 , 'stage5': 56,  'stage6': 112  }
    elif args.img_size == 128:
        patches = {'stage1': 4, 'stage2': 4*2, 'stage3': 4*2*2, 'stage4':32, 'stage5': 64, 'stage6': 128 }
    elif args.img_size == 64:
        patches = {'stage1': 4, 'stage2': 4*2, 'stage3': 4*2*2, 'stage4':32, 'stage5': 64}
    elif args.img_size == 32:
        patches = {'stage1': 4, 'stage2': 4*2, 'stage3': 4*2*2, 'stage4':32 }


    if stage != 'stage1':
        with open(args.mask_path+'.pkl', 'rb') as f:
            S1_mask = pickle.load(f)
    print('Number of candidates: %d' %num_candidates)
    for _ in range(int(num_candidates/3)):
         
        c = np.random.randint(args.num_class)
        M = np.random.randint(10)
        patch = patches[stage]
        freqs = gen_freqs_list(patch,patch)
        sample_frqs = sample_frequency(portion, freqs)
        mask_int = generate_mask(sample_frqs,patch,patch)
        mask = np.kron(mask_int, np.ones((int(args.img_size/patch), int(args.img_size/patch))))
        if stage != 'stage1':
            mask  = S1_mask[c][M]  * mask
        
        for h_index in range(-max_n_h, 1):   
            for w_index in range(-max_n_w, max_n_w ):
                h_matrix_index = int(np.floor(args.img_size / 2)) + h_index
                w_matrix_index = int(np.floor(args.img_size / 2)) + w_index
                if h_index != 0:
                    mask[args.img_size - h_matrix_index - 1, args.img_size - w_matrix_index - 1] = mask[h_matrix_index,w_matrix_index]
        # print(mask)
        DEFALUT_CANDIDATES.append(White_Mask(mask))
        if patches[stage] != args.img_size :
            patch = patches[stage]+1
            freqs = gen_freqs_list(patch,patch)
            sample_frqs = sample_frequency(portion, freqs)
            mask_int = generate_mask(sample_frqs,patch,patch)
            mask = np.kron(mask_int, np.ones((int(args.img_size/(patch-1)), int(args.img_size/(patch-1)))))
            mask = mask[int(args.img_size/(patch-1)/2):-int(args.img_size/(patch-1)/2),int(args.img_size/(patch-1)/2):-int(args.img_size/(patch-1)/2)]
            if stage != 'stage1':
                mask  = S1_mask[c][M]  * mask
            for h_index in range(-max_n_h, 1):   
                for w_index in range(-max_n_w, max_n_w ):
                    h_matrix_index = int(np.floor(args.img_size / 2)) + h_index
                    w_matrix_index = int(np.floor(args.img_size / 2)) + w_index
                
                    if h_index != 0:
                        mask[args.img_size - h_matrix_index - 1, args.img_size - w_matrix_index - 1] = mask[h_matrix_index,w_matrix_index]
            # print(mask)
            DEFALUT_CANDIDATES.append(White_Mask(mask))

            patch = patches[stage]
            freqs = gen_freqs_list(patch,patch)
            sample_frqs = sample_frequency(portion/2, freqs)
            mask_int = generate_mask(sample_frqs,patch,patch)
            mask_int1 = np.kron(mask_int, np.ones((int(args.img_size/patch), int(args.img_size/patch))))

            patch = patches[stage]+1
            freqs = gen_freqs_list(patch,patch)
            sample_frqs = sample_frequency(portion/2, freqs)
            mask_int = generate_mask(sample_frqs,patch,patch)
            mask_int2 = np.kron(mask_int, np.ones((int(args.img_size/(patch-1)), int(args.img_size/(patch-1)))))
            mask_int2 = mask_int2[int(args.img_size/(patch-1)/2):-int(args.img_size/(patch-1)/2),int(args.img_size/(patch-1)/2):-int(args.img_size/(patch-1)/2)]
            mask =np.clip( mask_int1+mask_int2, 0,1)
            if stage != 'stage1':
                mask = S1_mask[c][M]  * mask
            for h_index in range(-max_n_h, 1):   
                for w_index in range(-max_n_w, max_n_w ):
                    h_matrix_index = int(np.floor(args.img_size / 2)) + h_index
                    w_matrix_index = int(np.floor(args.img_size / 2)) + w_index
                
                    if h_index != 0:
                        mask[args.img_size - h_matrix_index - 1, args.img_size - w_matrix_index - 1] = mask[h_matrix_index,w_matrix_index]
            # print(mask)
            DEFALUT_CANDIDATES.append(White_Mask(mask))
    return DEFALUT_CANDIDATES




def validate_child(args, model, dataset, subset_indx, transform, device=None):
    criterion = nn.CrossEntropyLoss()

    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataset.transform = transform
    print(dataset.transform)
    subset = Subset(dataset, subset_indx)
    data_loader = get_dataloader(args, subset, pin_memory=False)

    return validate_classwise(args, model, criterion, data_loader, device)


def get_next_subpolicy(args,transform_candidates,b):
    n_candidates = len(transform_candidates)
    subpolicy = []
    only_subpolicy = []
  
    print(n_candidates)
    print(b) 
    
    indx = b
    subpolicy.append(transform_candidates[indx])
    only_subpolicy.append(transform_candidates[indx])
    if args.dataset == 'imagenet1k':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'imagenet10':
        mean = [0.479838, 0.470448, 0.429404]
        std = [0.258143, 0.252662, 0.272406]  
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif  args.dataset == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.243, 0.262]


    
    subpolicy = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),transforms.Normalize(mean, std),*subpolicy,])
 
    return subpolicy,only_subpolicy[0]


def search_subpolicies(args, transform_candidates, child_model, dataset, Da_indx, device):
    subpolicies = []
    best_loss = {}
    for b in range(len(transform_candidates)):
        subpolicy,only_subpolicy = get_next_subpolicy(args,transform_candidates,b)
        val_res = validate_child(args, child_model, dataset, Da_indx, subpolicy, device)
        if b == 0:
            for c in range(len(val_res[0])):
                best_loss.update({c:[val_res[0][c]]})
        else:
            for c in range(len(val_res[0])):
                if val_res[0][c] < best_loss[c][-1]:
                    best_loss[c].append(val_res[0][c])  
                else:
                     best_loss[c].append(best_loss[c][-1])  
        subpolicies.append((only_subpolicy, val_res[0])) 
    return subpolicies, best_loss



def get_topn_subpolicies_classwise(args, subpolicies, N=10):
    sub = []

    for cla in range(args.num_class):
        sub_c = sorted(subpolicies, key=lambda subpolicy: subpolicy[1][cla].cpu().numpy())[:N]
        sub.append(sub_c)


    return sub

def process_fn(args_str, model, dataset, Da_indx, transform_candidates,  N, k):
    kwargs = json.loads(args_str)
    args, kwargs = parse_args(kwargs)
    device_id = k % torch.cuda.device_count()
    device = torch.device('cuda:%d' % device_id)
    _transform = {}
    
 
    subpolicies , loss_track = search_subpolicies(args, transform_candidates, model, dataset, Da_indx, device)
    subpolicies = get_topn_subpolicies_classwise(args,subpolicies, N)
    for c in range(args.num_class):
        _transform.update({c:[subpolicy[0] for subpolicy in subpolicies[c]]})
    with open("loss_track/"+args.maskname+".pkl", 'wb') as f:
        pickle.dump(loss_track, f)
    f.close()

    return _transform


def LFSI(args, model, transform_candidates=None, K=1,  N=10, num_process=1):
    args_str = json.dumps(args._asdict())
    if args.dataset == 'imagenet1k':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'imagenet10':
        mean = [0.479838, 0.470448, 0.429404]
        std = [0.258143, 0.252662, 0.272406]  
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif  args.dataset == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.243, 0.262]


    standard_transform =  transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),transforms.Normalize(mean, std)])
    dataset = get_dataset(args, standard_transform)
    num_process = num_process  
    futures = []

    torch.multiprocessing.set_start_method('spawn', force=True)
    
   
    print(args.stage)
    DEFALUT_CANDIDATES =  get_candidates(args,num_candidates=args.N_B,portion=args.P,stage=args.stage)
    transform_candidates = DEFALUT_CANDIDATES

    _, Da_indexes = split_dataset(args, dataset, K)

    for k, (Da_indx) in enumerate(Da_indexes):
        future = process_fn(args_str, model, dataset,  Da_indx, transform_candidates, N, k)
        futures.append(future)
   

    
   
    masks = {}
    for c in range(args.num_class):
        masks.update({c: [t.mask for t in futures[0][c]]})

    return  masks