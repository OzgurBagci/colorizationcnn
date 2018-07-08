from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2lab, lab2rgb
import torch
import torch.nn.functional as F
import numpy as np
import os


def read_imgs(path, img_names, gray, for_cnn=False):
    """
    Converts images to CIE LAB format.

    Note that path should end with '/' char.

    :param path: str
    :param img_names: list(str)
    :param gray: bool
    :param for_cnn: bool
    :return: np.array
    """

    images = []
    for img_name in img_names:
        img = imread(path + img_name, as_grey=gray)
        if not gray:
            img = rgb2lab(img)
        else:
            img = np.expand_dims(img_as_float(img), 2)
        images.append(img)
    images = np.array(images)
    if for_cnn:
        images = images[:, :, :, (1, 2)].astype(float) / 128
    return images


def write_npy(path, filename, arr):
    """
    Given arr must contain images in CIE LAB format.

    Note that path should end with '/' char.

    :param path: str
    :param filename: str
    :param arr: np.array
    :return: None
    """

    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + filename, arr.astype(int))


def create_rgb(path, txt, ab, is_cuda=False):
    """
    Converts AB output and L input to RGB images. AB must be 64x64 tensor with 2 channels and L will be read from path.

    :param path: str
    :param txt: str
    :param ab: torch.Tensor
    :param is_cuda: bool
    :return: torch.Tensor
    """
    if is_cuda:
        ab = ab.to(torch.device('cpu'))

    with open(txt) as vf:
        lines = vf.readlines()

    filenames = []
    for line in lines:
        filenames.append(line[:-1])

    l = read_imgs(path, filenames, True) * 100
    ab = F.upsample(ab, size=(256, 256), mode='bilinear').detach().numpy() * 128
    ab = np.einsum('ijkl->iklj', ab)
    lab = np.concatenate((l[0:len(ab)], ab), axis=3)
    rgb = []
    for im in lab:
        rgb.append(img_as_float(lab2rgb(im)) * 255)
    rgb = torch.Tensor(rgb)
    return rgb
