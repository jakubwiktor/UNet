from __future__ import absolute_import, division
import numpy as np
import torch

#to be used with albumentations lambda transformation
def custom_normalize(image, **kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    image = (image - np.mean(image)) / np.std(image)
    return image

def custom_to_tensor(image, **kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    this_type = torch.FloatTensor
    # this_type = torch.HalfTensor #16-bit floating point
    image = torch.from_numpy(image).type(this_type)
    image = image.unsqueeze(0)
    return image

def custom_gauss_noise(image,**kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    mean = 0
    var = np.random.uniform((2**16)/4,(2**16)*4)
    sigma = var ** 0.5
    random_state = np.random.RandomState(np.random.randint(0, 2 ** 32 - 1))
    gauss = random_state.normal(mean, sigma, image.shape)

    return image + gauss

def main():
    pass

if __name__ == '__main__':
    main()