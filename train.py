coding='utf-8'
import os
from net import Mynet
import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import cv2
from eval import PR_Curve,MAE_Value
from smooth_loss import get_saliency_smoothness,get_grid_smoothness
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def myloss(scores,label):
    deepsal_loss = F.binary_cross_entropy(torch.sigmoid(scores[0]),label,reduction='mean')+\
                   F.binary_cross_entropy(torch.sigmoid(scores[1]),label,reduction='mean')+\
                   F.binary_cross_entropy(torch.sigmoid(scores[2]),label,reduction='mean')+ \
                   F.binary_cross_entropy(torch.sigmoid(scores[3]), label, reduction='mean') + \
                   F.binary_cross_entropy(torch.sigmoid(scores[4]),label,reduction='mean')
    return deepsal_loss

if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    
    # dataset
    img_root = './data/VT5000-Train_unalign/'
    save_path = './model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not os.path.exists(temp_save_path): os.mkdir(temp_save_path)
    lr = 0.001 #2
    batch_size = 4
    epoch = 100
    lr_dec=[51]
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

    net = Mynet().cuda()
    net.load_pretrained_model()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)

    iter_num = len(loader)
    net.train()
    for epochi in range(1, epoch + 1): 
    
        if epochi in lr_dec :
            lr=lr/10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)
            print(lr)
        prefetcher = DataPrefetcher(loader)
        rgb, t, label = prefetcher.next()
        r_sal_loss = 0
        net.zero_grad()
        i = 0
        while rgb is not None:
            i+=1
            scores = net(rgb, t)
            loss = myloss(scores, label)
            r_sal_loss += loss.data
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100))
                r_sal_loss = 0
            rgb, t, label = prefetcher.next()
    torch.save(net.state_dict(), '%s/final.pth' % (save_path))