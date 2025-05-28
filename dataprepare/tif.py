# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import os

# Parameters
height = 512  # 模型输入图像高度
width  = 512  # 模型输入图像宽度
channels = 3  # 图像通道数

train_number = 1  # 训练集样本数
val_number = 1    # 验证集样本数
test_number = 1   # 测试集样本数
all = int(train_number) + int(val_number) + int(test_number)

#############################################################
# 准备数据集
# 图像存放在 "./data/FIO-EP/image/" 文件夹中（tif格式），
# 标签存放在 "./data/FIO-EP/label/" 文件夹中（tif格式）。
#############################################################
Tr_list = glob.glob("./data/one/train/images/*.tif")  # 匹配 tif 格式文件
# 初始化数据数组（确保样本总数不超过文件夹中的图像数）
Data_train_2018  = np.zeros([all, height, width, channels])
Label_train_2018 = np.zeros([all, height, width])

print('Reading')
print("Total images:", len(Tr_list))
for idx in range(len(Tr_list)):
    print("Processing image:", idx+1)
    # 读取原始图像，并转换为RGB格式
    img = Image.open(Tr_list[idx]).convert('RGB')
    img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img).astype(np.float64)
    Data_train_2018[idx, :, :, :] = img

    # 提取图像文件名（不包含扩展名），构造标签文件名（tif格式）
    base_name = os.path.splitext(os.path.basename(Tr_list[idx]))[0]
    label_file = base_name + ".tif"
    label_path = os.path.join("./data/one/train/label/", label_file)
    
    # 读取标签图像，转换为灰度图
    img2 = Image.open(label_path).convert('L')
    # 使用最近邻插值，适用于离散标签
    img2 = img2.resize((width, height), Image.NEAREST)
    img2 = np.array(img2).astype(np.float64)
    Label_train_2018[idx, :, :] = img2    

print('Reading your dataset finished')

################################################################
# 划分训练集、验证集和测试集
################################################################
Train_img      = Data_train_2018[0:train_number, :, :, :]  
Validation_img = Data_train_2018[train_number:train_number+val_number, :, :, :]
Test_img       = Data_train_2018[train_number+val_number:all, :, :, :]

Train_mask      = Label_train_2018[0:train_number, :, :]
Validation_mask = Label_train_2018[train_number:train_number+val_number, :, :]
Test_mask       = Label_train_2018[train_number+val_number:all, :, :]

np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)
