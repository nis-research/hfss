import os
import time
import collections
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import pickle
from sklearn.model_selection import StratifiedShuffleSplit


DATASET_PATH = '../datasets/'
current_epoch = 0


def split_dataset(args, dataset, k, chosen_class=None):
    # load dataset
    X = list(range(len(dataset)))
    Y = dataset.targets

    # split to k-fold
    assert len(X) == len(Y)

    def _it_to_list(_it):
        return list(zip(*list(_it)))

  
    ts = args.ts
    sss = StratifiedShuffleSplit(n_splits=k, random_state=args.seed, test_size=ts)
    Dm_indexes, Da_indexes = _it_to_list(sss.split(X, Y))
    Da_indexes  = list(Da_indexes)
    print(Da_indexes)
    print(len(Da_indexes[0]))
    if isinstance(chosen_class, int):
        for n_split in range(k):
            Da_indexes[n_split] = [d for d in Da_indexes[n_split] if Y[d] == chosen_class]

    Da_indexes = tuple(Da_indexes)
    return Dm_indexes, Da_indexes


def concat_image_features(image, features, max_features=3):
    _, h, w = image.shape

    max_features = min(features.size(0), max_features)
    image_feature = image.clone()

    for i in range(max_features):
        feature = features[i:i+1]
        _min, _max = torch.min(feature), torch.max(feature)
        feature = (feature - _min) / (_max - _min + 1e-6)
        feature = torch.cat([feature]*3, 0)
        feature = feature.view(1, 3, feature.size(1), feature.size(2))
        feature = F.upsample(feature, size=(h,w), mode="bilinear")
        feature = feature.view(3, h, w)
        image_feature = torch.cat((image_feature, feature), 2)

    return image_feature


def get_model_name(args):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%B_%d_%H:%M:%S")
    model_name = '__'.join([ args.network, str(args.seed)])
    return model_name


def dict_to_namedtuple(d):
    Args = collections.namedtuple('Args', sorted(d.keys()))

    for k,v in d.items():
        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    args = Args(**d)
    return args


def parse_args(kwargs):
    # combine with default args
    kwargs['dataset'] =  kwargs['dataset'] if 'dataset' in kwargs else 'cifar10'
    kwargs['seed'] =  42
    kwargs['use_cuda'] =  kwargs['use_cuda'] if 'use_cuda' in kwargs else True
    kwargs['use_cuda'] =  kwargs['use_cuda'] and torch.cuda.is_available()
    kwargs['num_workers'] = kwargs['num_workers'] if 'num_workers' in kwargs else 4
    kwargs['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 4000
    kwargs['maskname'] = kwargs['maskname'] if 'maskname' in kwargs else 'mask'
    kwargs['mask_path'] = kwargs['mask_path'] if 'mask_path' in kwargs else 'masks'
    kwargs['model_path'] = kwargs['model_path'] if 'model_path' in kwargs else 'resnet18'
    kwargs['num_class'] = kwargs['num_class'] if 'num_class' in kwargs else 10
    kwargs['P'] = kwargs['P'] if 'P' in kwargs else 0.8
    kwargs['N_B'] = kwargs['N_B'] if 'N_B' in kwargs else 500
    kwargs['stage'] = kwargs['stage'] if 'stage' in kwargs else 'stage1'
  
    kwargs['ts'] = kwargs['ts'] if 'ts' in kwargs else 0.05
    kwargs['img_size'] = kwargs['img_size'] if 'img_size' in kwargs else 224
    args = dict_to_namedtuple(kwargs)
    print("kwargs: ",kwargs)
    print("args: ",args)
    return args, kwargs



def get_dataset(args, transform):
    data_root = '/local/swang'
    if args == None:
        dataset  = ImageFolder(os.path.join(data_root,'ImageNet','train'),transform=transform)
    elif args.dataset == 'imagenet10':
        dataset  = ImageFolder(os.path.join(data_root,'ImageNet10','train'),transform=transform)
    elif args.dataset == 'imagenet1k' :
        dataset  = ImageFolder(os.path.join(data_root,'ImageNet','train'),transform=transform)
    elif args.dataset == 'fgsm':
        dataset =  ImageFolder(os.path.join(data_root,'ImageNet','val') ,transform=transform)      
    elif args.dataset == 'imagenet_r':
        dataset = ImageFolder(os.path.join(data_root,"imagenet-r"), transform=transform)
    elif args.dataset == 'imagenet200':
        dataset = ImageFolder(os.path.join(data_root,"imagenet_val_for_imagenet_r"), transform=transform)   
    elif args.dataset == 'ImageNet_C':
        dataset =  ImageFolder(os.path.join(data_root,'ImageNet_C',args.corruption,  args.severity),transform=transform)
    elif args.dataset == 'sketch':
        dataset =  ImageFolder(os.path.join(data_root,'sketch'),transform=transform)   
    elif args.dataset == 'imagenet_match':
        dataset = ImageFolder(os.path.join(data_root,'imagenetv2-matched-frequency-format-val'),transform=transform)
    elif args.dataset == 'imagenet_thre':
        dataset = ImageFolder(os.path.join(data_root,'imagenetv2-threshold0.7-format-val'),transform=transform)
    elif args.dataset == 'imagenet_top':
        dataset = ImageFolder(os.path.join(data_root,'imagenetv2-top-images-format-val'),transform=transform)

    else:
        raise Exception('Unknown dataset')

    return dataset

def get_meanstd():
   
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  
    return mean, std

def get_dataloader(args, dataset, shuffle=False, pin_memory=True):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              num_workers=args.num_workers,
                                              pin_memory=pin_memory)
    return data_loader



def find_SI(args, model ):
   
    from LSI import LFSI 
    masks = LFSI(args, model)
    if not os.path.exists('./DFM/'):
        os.makedirs('DFM/')
    with open("./DFM/"+args.maskname+".pkl", 'wb') as f:
        pickle.dump(masks, f)
    f.close()



def validate_classwise(args, model, criterion, valid_loader,  device=None):
    # switch to evaluate mode
    model.eval()
    if args.dataset == 'imagenet10' :
        loss = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    elif args.dataset == 'imagenet1k':
        loss = {}
        for cla in range(1000):
            loss.update({cla:0})
    
    infer_t = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):

            start_t = time.time()
            if device:
                images = images.to(device)
                target = target.to(device)

            elif args.use_cuda is not None:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            for c in loss:
                reference_class = torch.ones(target.size(0))* c
                tested_classes = (target == reference_class.to(device))
                # tested_classes = tested_classes.int()
                
                tc = torch.unsqueeze(tested_classes,1)
                TC = [tc,tc,tc,tc,tc,tc,tc,tc,tc,tc]
                if args.dataset == 'imagenet1k':
                    for _ in range(990):
                        TC.append(tc)

                test_cla = torch.cat(tuple(TC),1).to(device)
            
                l_c = criterion(test_cla*output,tested_classes*target)
                loss[c] += l_c

            infer_t += time.time() - start_t

        


    return loss, infer_t



# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)