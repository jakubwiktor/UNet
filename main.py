from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

from utils.custom_loader import custom_loader_training
from utils.mm_classifier import mm_classifier
from utils.custom_a_handles import custom_normalize, custom_to_tensor, custom_gauss_noise

#using albumentations library
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

# from utils.unet import UNet
from utils.unet import UNet

from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.load_files import getFileList

# from supporting_functions.loadDataSets import mmDataSetTrain
# import supporting_functions.customTransformations as ct
# from supporting_functions.mmClassifier import mmClassifier


def train_net(save_name = None):

    #load the data
    phase_dirs =[
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/GroundTruths/exp4_agarosePad_phase/', ''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/exp3_agarosePad_phase/',''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/Kuba/','kuba_raw'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/Praneeth/','P_raw'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BV3242/exp7_phase',''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/empty_MM_channels/','aphase'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BT2878/','raw')
                 ]
    
    phase = [getFileList(dr,nm) for dr, nm in phase_dirs]
    phase = [this_phase for each_set in phase for this_phase in each_set]

    mask_dirs = [
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/GroundTruths/exp4_agarosePad_gt/', ''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/exp3_agarosePad_gt/',''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/Kuba/','kuba_gt'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/Praneeth/','P_gt'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BV3242/exp7_gt',''),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BQ2462/empty_MM_channels/','gt'),
                 ('/hdd/RecPAIR/Microscopy/EXP-20-BQ2489/BT2878/','gt')
                 ]

    mask = [getFileList(dr,nm) for dr, nm in mask_dirs]
    mask = [this_mask for each_set in mask for this_mask in each_set]
   
    # phase = phase[1:10]
    # mask = mask[1:10]
    print(len(phase))
    print(len(mask))

    #specify transformations and unets
    crop_window = 512

    #use albumentations library
    training_transform = A.Compose(
        [   
            A.RandomRotate90(p=1),
            A.RandomCrop(crop_window, crop_window, p=1),
            A.GridDistortion(p=0.6),
            A.ElasticTransform(p=0.6, alpha=100, sigma=200 * 0.05, alpha_affine=200 * 0.03),
            A.ShiftScaleRotate(p=1, shift_limit=0.2, scale_limit=0.25, rotate_limit=45),
            A.GaussianBlur(p=0.6, blur_limit=(3, 7), sigma_limit=0),
            A.Lambda(name='gauss-noise', image=custom_gauss_noise, p=0.5),
            A.GridDropout(p=0.5, ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=True, fill_value=0, mask_fill_value=None),
            A.Lambda(name='normalize', image=custom_normalize, p=1.0),
            A.Lambda(name='to_tensor', image=custom_to_tensor, mask=custom_to_tensor, p=1.0)
        ]
        )

    training_dataset = custom_loader_training( phase_ims = phase, 
                                               mask_ims  = mask,
                                               transform = training_transform)

    validate_dataset = custom_loader_training(phase_ims = phase, 
                                              mask_ims = mask, 
                                              transform = A.Compose([A.RandomCrop(512, 512,p=1),
                                                                    A.Lambda(name='normalize', image=custom_normalize, p=1.0),
                                                                    A.Lambda(name='to_tensor', image=custom_to_tensor, mask=custom_to_tensor, p=1.0),
                                                                     ]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet()
    # net = UNet1024()
    net.cuda()
    
    training_loader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers = 10)
    validation_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

    # quick test
    # for ti, (im,mask) in enumerate(training_loader):
    #         im = im.numpy()[0][0]
    #         print('next im')
    #         print(np.min(im))
    #         print(np.max(im))
    #         im = im - np.min(im)
    #         im = im / np.max(im)
    #         mask = mask.numpy()[0].astype('uint16')*(2**16-1)
    #         # blend = cv2.addWeighted(im,1,mask,0.5,0)
    #         cv2.imshow('blend',im)
    #         cv2.waitKey(500)
    #         if ti > 10:
    #             break
    # return

    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    # scheduler = None
    num_epochs = 30
    
    classifier = mm_classifier(net=net, 
                               optimizer = optimizer, 
                               scheduler = scheduler, 
                               num_epochs = num_epochs, 
                               save_name = save_name)

    classifier.train(training_loader, validation_loader)
 
def main():
    #fill in the name
    NET_NAME = 'UNet_large'
    save_name = f"/hdd/RecPAIR/{NET_NAME}.pth"
    train_net(save_name)

if __name__ == '__main__':
    main()