from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import prune
from torchvision import transforms
import torch

import os

from utils.unet import UNet

from skimage import io, measure, morphology, feature, color, transform
import matplotlib.pyplot as plt
import numpy as np


def segment():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet()

    NET_PATH = '/hdd/RecPAIR/Unet_universal.pth'

    saved_net = torch.load(NET_PATH)
    net.load_state_dict(saved_net['model_state_dict'])
    net.eval()
    net.cuda()

    dir_name = '/home/skynet/code/UNet_2021/test_cases/'
    test_cases = os.listdir(dir_name)
    dir_name = '/hdd/05 El330/O2/Pos3/phase'

    test_cases = ['img_000000000.tiff']

    #load, normalize and predict for each image in the test directory
    for fi, f in enumerate(test_cases):
        print(os.path.join(dir_name, f))
        
        im_org = io.imread(os.path.join(dir_name, f))
        im_org = transform.rescale(im_org, 0.5)
        print(im_org.shape)
        im_org = im_org[0:1200,0:1200]
        im = im_org.astype('float32')
        im = (im - np.mean(im)) / np.std(im)
        im = torch.from_numpy(im)
        im = im.unsqueeze(0).unsqueeze(0)
        im = im.cuda()
        res = net(im)
        res = torch.sigmoid(res)
        res = res.to("cpu").detach().numpy().squeeze(0).squeeze(0)
        
        res = res > 0.5
        
        res = morphology.remove_small_objects(res,50)
        outlines = feature.canny(res)

        
        plt.figure()

        vmin = np.percentile(im_org,5)
        vmax = np.percentile(im_org,95)
        
        im_org = color.gray2rgb(im_org)
        im_org = im_org - np.min(im_org)
        im_org = im_org / np.max(im_org)

        for yi,y in enumerate(outlines):
            for xi,x in enumerate(y):
                if x:
                    im_org[yi][xi] = (0,1,0)

        plt.imshow(im_org,cmap=plt.cm.gray)
        plt.show()

def pruneNet():
    
    NET_PATH = '/hdd/RecPAIR/UNet_Praneeth.pth'
    net = torch.load(NET_PATH)
    net = net['model_state_dict']
    torch.save(net,'/hdd/RecPAIR/UNet_Praneeth_net.pth')

def main():
    segment()
    # pruneNet()

if __name__ == '__main__':
    main()
