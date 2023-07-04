import os
import random

import cv2
import shutil
from tqdm import tqdm




def copy_data():
    # copy_path = r'/project/SphereFace2Data/real_train_sphereface2_300w.txt'
    # shutil.copy(copy_path,'/img_data4')
    img_dir = r'/project/SphereFace2Data/SocialSecurityAligned'
    list1 = os.listdir(img_dir)
    for target in tqdm(list1):
        if '.zip' in list1:
            continue
        img_path = os.path.join(img_dir,target)
        for path in os.listdir(img_path):
            move_path = os.path.join(img_path,path)
            target = os.path.join('/project/SphereFace2Data/SocialSecurityAligned_For_Test',path)
            shutil.copytree(move_path,target)


def resize_img(img_dir):
    all_img_path = []
    for root,dirs,files in os.walk(img_dir):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_path = os.path.join(root,filename)
                all_img_path.append(img_path)

    for img_p in tqdm(all_img_path):
        img = cv2.imread(img_p)
        img = cv2.resize(img,(128,128))
        save_path = img_p.replace('DigiFace1M','DigiFace1M_128x128')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        cv2.imwrite(save_path,img)

def pick_less2(image_dir):
    all_id = list(os.listdir(image_dir))
    move_path = r'/img_data4/zjk_pick/part_17'
    os.makedirs(move_path,exist_ok=True)
    num = 1
    for id in tqdm(all_id):
        image_path = os.path.join(image_dir, id)
        if num <= 100000:
            shutil.move(image_path, move_path)
            num += 1


def get_info(img_dir):
    all_num = 0
    imgs_list = os.listdir(img_dir)
    print(len(imgs_list))
    for id in tqdm(imgs_list):
        img_dirs = os.path.join(img_dir,id)
        all_num += len(os.listdir(img_dirs))
    print(all_num)

def get_info2(img_dir):
    all_num = 0
    classes_num = 0
    img_list = os.listdir(img_dir)
    for first in tqdm(img_list):
        if 'zip' in first:
            continue
        img_dir2 = os.path.join(img_dir,first)
        img_list2 = os.listdir(img_dir2)
        classes_num += len(img_list2)
        for two in tqdm(img_list2):
            img_dir3 = os.path.join(img_dir2,two)
            all_num += len(os.listdir(img_dir3))
    print(all_num)
    print(classes_num)

def pick_for_test(img_dir):
    img_list = os.listdir(img_dir)
    pick_id = random.sample(range(0,len(img_list)),2000)
    for id in tqdm(pick_id):
        pick_path = os.path.join(img_dir,img_list[id])
        tmp = img_dir.replace('train_merge','val_merge',1)
        move_path = os.path.join(tmp,img_list[id])
        shutil.move(pick_path,move_path)

def get_origin_data(img_dir):
    img_list = os.listdir(img_dir)
    for index,id in tqdm(enumerate(img_list)):
        origin_dir = os.path.join(img_dir,id).replace('cc_align_test','cc_data_1')
        copy_path = os.path.join('/img_data4/cc_test_origin',str(index))
        os.makedirs(copy_path,exist_ok=True)
        img_dir2 = os.path.join(img_dir,id)
        for image in os.listdir(img_dir2):
            origin_image_path = os.path.join(origin_dir,image)
            shutil.copy(origin_image_path,copy_path)

def rename_img(img_dir):
    img_list = os.listdir(img_dir)
    for index, id in tqdm(enumerate(img_list)):
        image_dir = os.path.join(img_dir,id)
        new_name = os.path.join(img_dir,str(index))
        os.rename(image_dir,new_name)


if __name__ == '__main__':
    img_dir = r'/project/SphereFace2Data/SocialSecurityAligned'
    # get_origin_data(img_dir)
    get_info2(img_dir)
    # pick_less2(img_dir)
    # rename_img(img_dir)