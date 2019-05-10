# -*- coding: utf-8 -*
"""
Created by Xingxiangrui on 2019.5.9
This code is to :
    1. copy image from source_image_dir to the target_image_dir
    2. And generate .txt file for further training
        in which each line is : image_name.jpg  (tab)  image_label (from 0)
        such as:
            image_01.jpg    0
            iamge_02.jpg    1
            ...
            image_02.jpg    0

"""



# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random

# variables need to be change
source_image_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/single-poly-defect/poly_OK"
target_image_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/data_for_resnet_classification"
txt_file_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/data_for_resnet_classification/TxtFile"
prefix="poly_OK"
class_label=1
# label 0: single_OK ; label_1: poly_OK ; label 2: poly_defect

print("Program Start......")
print("-"*20)
print("-"*20)
print("-"*20)

# load image list in the source dir
source_image_list = os.listdir(source_image_dir)
for idx in range(len(source_image_list)):
    if '.png' in source_image_list[idx-1]:
        continue
    elif '.jpg' in source_image_list[idx-1]:
        continue
    else:
        del source_image_list[idx-1]

# shuffle image list
print("initial list:")
print source_image_list
random.shuffle(source_image_list)
print("shuffled list:")
print source_image_list

# train list and val list
source_train_list=[]
source_val_list=[]
for idx in range(len(source_image_list)):
    if idx<len(source_image_list)/4:
        source_val_list.append(source_image_list[idx-1])
    else:
        source_train_list.append(source_image_list[idx-1])
print ("train_list")
print source_train_list
print("val_list")
print source_val_list

# create label_file or write label file
txt_file_train_name="train.txt"
txt_file_val_name="val.txt"
txt_file_train_path=os.path.join(txt_file_dir, txt_file_train_name)
txt_file_val_path=os.path.join(txt_file_dir, txt_file_val_name)
train_txt_file= open(txt_file_train_path,"a")
val_txt_file= open(txt_file_val_path,"a")

# write train images and labels
print("write train images and labels......")
for source_image_name in source_train_list:
    print source_image_name

    # read dource images and rename
    path_source_img = os.path.join(source_image_dir, source_image_name)
    src_img = Image.open(path_source_img)
    full_image_name=prefix+"_train_"+source_image_name
    print(full_image_name)
    # save renamed image to the target dir
    target_image_path=os.path.join(target_image_dir, full_image_name)
    src_img.save(target_image_path)
    # write image names and labels
    line_strings= full_image_name+"\t"+str(class_label)+"\n"
    train_txt_file.write(line_strings)

# write val images and labels
print("write val images and labels......")
for source_image_name in source_val_list:
    print source_image_name

    # read dource images and rename
    path_source_img = os.path.join(source_image_dir, source_image_name)
    src_img = Image.open(path_source_img)
    full_image_name=prefix+"_val_"+source_image_name
    print(full_image_name)
    # save renamed image to the target dir
    target_image_path=os.path.join(target_image_dir, full_image_name)
    src_img.save(target_image_path)
    # write image names and labels
    line_strings= full_image_name+"\t"+str(class_label)+"\n"
    val_txt_file.write(line_strings)

print("source_image_dir:")
print source_image_dir
print("target_image_dir:")
print target_image_dir
print("prefix:")
print prefix
print("label:")
print class_label

print("image numbers:")
print len(source_image_list)





'''
import numpy as np
from PIL import Image
import os
import random

# variables need to be change
source_image_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/single-poly-defect/poly_defect_gen"
target_image_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/data_for_resnet_classification"
txt_file_dir="/Users/baidu/Desktop/used/SuZhouRuiTu_dataset/data_for_resnet_classification/TxtFile"
prefix="gen_poly_defect"
class_label=2
# label 0: single_OK ; label_1: poly_OK ; label 2: poly_defect

print("Program Start......")
print("-"*20)
print("-"*20)
print("-"*20)

# load image list in the source dir
source_image_list = os.listdir(source_image_dir)
for idx in range(len(source_image_list)):
    if '.png' in source_image_list[idx-1]:
        continue
    elif '.jpg' in source_image_list[idx-1]:
        continue
    else:
        del source_image_list[idx-1]



# create label_file or write label file
txt_file_train_name="train.txt"
# txt_file_val_name="val.txt"
txt_file_train_path=os.path.join(txt_file_dir, txt_file_train_name)
# txt_file_val_path=os.path.join(txt_file_dir, txt_file_val_name)
train_txt_file= open(txt_file_train_path,"a")
# val_txt_file= open(txt_file_val_path,"a")

# write train images and labels
print("write train images and labels......")
for source_image_name in source_image_list:
    print source_image_name

    # read dource images and rename
    path_source_img = os.path.join(source_image_dir, source_image_name)
    src_img = Image.open(path_source_img)
    full_image_name=prefix+"_train_"+source_image_name
    print(full_image_name)
    # save renamed image to the target dir
    target_image_path=os.path.join(target_image_dir, full_image_name)
    src_img.save(target_image_path)
    # write image names and labels
    line_strings= full_image_name+"\t"+str(class_label)+"\n"
    train_txt_file.write(line_strings)


print("source_image_dir:")
print source_image_dir
print("target_image_dir:")
print target_image_dir
print("prefix:")
print prefix
print("label:")
print class_label

print("image numbers:")
print len(source_image_list)


'''












