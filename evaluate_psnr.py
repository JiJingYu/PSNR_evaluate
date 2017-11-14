from __future__ import print_function
import argparse
from math import log10

import torch
import time
import h5py
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
from copy import deepcopy
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import FloatTensor
from sklearn.model_selection import train_test_split
from torchvision import transforms
from accuracy_PSNR import mpsnr, mnrmse, mrmse, mnrmse_v2
from skimage.measure import compare_mse
import matplotlib.pyplot as plt
from scipy import misc
from skimage.measure import compare_psnr, compare_ssim, compare_mse, compare_nrmse


def load_image(factor = 3):
    img = plt.imread('./data/balloons_ms_01.png')
    im_true = img * 65535.0
    H, W = im_true.shape
    H -= H % factor
    W -= W % factor
    im_true = im_true[:H, :W]
    lr_img = misc.imresize(im_true, size=1.0 / factor, interp='bicubic', mode="F")
    return im_true, lr_img


def exp_001():
    # 截断与不截断
    print("############################\n")
    print("一、截断操作对性能的影响")
    factor = 3
    im_true, lr_img = load_image(factor = factor)

    hr_img = misc.imresize(lr_img, size= factor/1.0 ,interp='bicubic', mode="F")
    psnr_value = compare_psnr(im_true=im_true, im_test=hr_img, data_range=np.max(im_true))
    print("不截断，PSNR value is : {:.2f}".format(psnr_value))
    hr_img[hr_img < 0] = 0
    psnr_value = compare_psnr(im_true=im_true, im_test=hr_img, data_range=np.max(im_true))
    print("截断，PSNR value is : {:.2f}".format(psnr_value))


def exp_002():
    # 是否转换为uint16 对性能的影响
    print("############################\n")
    print("二、是否转换为uint16对性能的影响")
    factor = 3
    im_true, lr_img = load_image(factor=factor)

    hr_img = misc.imresize(lr_img, size=factor / 1.0, interp='bicubic', mode="F")
    psnr_value = compare_psnr(im_true=im_true.astype(np.uint16), im_test=hr_img.astype(np.uint16), data_range=np.max(im_true))
    print("1. 不截断，直接转换为uint16, \n"
          "PSNR value is : {:.2f}".format(psnr_value))
    hr_img[hr_img < 0] = 0
    psnr_value = compare_psnr(im_true=im_true.astype(np.uint16), im_test=hr_img.astype(np.uint16), data_range=np.max(im_true))
    print("2. 截断，直接转换为uint16, \n"
          "PSNR value is : {:.2f}".format(psnr_value))


def exp_003():
    # 动态范围如何设置
    print("############################\n")
    print("三、动态范围如何设置对性能的影响\n"
          "实验条件：先截断，再转换为uint16")
    factor = 3
    im_true, lr_img = load_image(factor=factor)

    hr_img = misc.imresize(lr_img, size=factor / 1.0, interp='bicubic', mode="F")
    hr_img[hr_img < 0] = 0
    psnr_value = compare_psnr(im_true=im_true.astype(np.uint16), im_test=hr_img.astype(np.uint16),
                              data_range=np.max(im_true))
    print("1. 动态范围设置为ground truth的最大值, \n"
          "PSNR value is : {:.2f}\n".format(psnr_value))

    psnr_value = compare_psnr(im_true=im_true.astype(np.uint16), im_test=hr_img.astype(np.uint16),
                              data_range=65535)
    print("2. 动态范围设置为65535, \n"
          "PSNR value is : {:.2f}\n".format(psnr_value))

    scale = np.max(im_true)
    im_true /= scale
    hr_img /= scale
    psnr_value = compare_psnr(im_true=im_true, im_test=hr_img, data_range=1)
    print("3. 动态范围设置为1\n"
          "Ground Truth 缩放至0-1区间, 输出图像同比缩放。\n"
          "PSNR value is : {:.2f}".format(psnr_value))


if __name__=="__main__":
    exp_001()
    exp_002()
    exp_003()