import cv2
import random
import numpy as np
from PIL import ImageEnhance


# global parameter
set_ratio = 0.5


# ===================== deformable data augmentation for input image =====================
def data_aug_flip(image, mask):
    if random.random()<set_ratio:
        return image, mask, False
    return image[:,::-1,:], mask[:,::-1], True


# ===================== texture data augmentation for input image =====================   
def data_aug_blur(image):
    if random.random()<set_ratio:
        return image
    
    select = random.random()
    if select < 0.3:
        kernalsize = random.choice([3,5])
        image = cv2.GaussianBlur(image, (kernalsize,kernalsize),0)
    elif select < 0.6:
        kernalsize = random.choice([3,5])
        image = cv2.medianBlur(image, kernalsize)
    else:
        kernalsize = random.choice([3,5])
        image = cv2.blur(image, (kernalsize,kernalsize))
    return image

def data_aug_color(image):  
    if random.random()<set_ratio:
        return image
    random_factor = np.random.randint(4, 17) / 10. 
    color_image = ImageEnhance.Color(image).enhance(random_factor) 
    random_factor = np.random.randint(4, 17) / 10. 
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(6, 15) / 10. 
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

def data_aug_noise(image):
    if random.random()<set_ratio:
        return image
    mu = 0
    sigma = random.random()*10.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    image[image>255] = 255
    image[image<0] = 0
    return image
