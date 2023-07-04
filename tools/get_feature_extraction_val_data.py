import os
import random
import shutil
import numpy as np
from tqdm import tqdm

def get_pos_num(image_path,save_txt_path,pos_num):
    save_txt = save_txt_path
    os.makedirs(os.path.dirname(save_txt), exist_ok=True)
    dir_id_list = list(os.listdir(image_path))
    all_save_line = []
    all_use_label = []
    for i in tqdm(range(0,pos_num)):
        dir_target = np.random.randint(0,len(dir_id_list),1)
        img_list = list(os.listdir(os.path.join(image_path,dir_id_list[dir_target[0]])))
        if len(img_list) <= 1:
            continue
        two_index = random.sample(range(0, len(img_list)), 2)
        img1, img2 = img_list[two_index[0]], img_list[two_index[1]]
        #防止产生重复标签对
        label1,label2 = os.path.join(dir_id_list[dir_target[0]],img1),os.path.join(dir_id_list[dir_target[0]],img2)
        while (label1,label2) in all_use_label:
            dir_target = np.random.randint(0, len(dir_id_list), 1)
            img_list = list(os.listdir(os.path.join(image_path, dir_id_list[dir_target[0]])))
            if len(img_list) <= 1:
                continue
            two_index = random.sample(range(0, len(img_list)),2)
            img1, img2 = img_list[two_index[0]], img_list[two_index[1]]
            label1, label2 = os.path.join(dir_id_list[dir_target[0]], img1), os.path.join(dir_id_list[dir_target[0]],img2)

        all_use_label.append((label1,label2))
        all_use_label.append((label2,label1))
        if img1.endswith(('.jpg', '.png', '.JPG', '.jpeg')) and img2.endswith(('.jpg', '.png', '.JPG', '.jpeg')):
            save_dir1, save_dir2 = os.path.join(image_path.split('/')[-1], dir_id_list[dir_target[0]], img1).replace('\\', '/'), os.path.join(image_path.split('/')[-1], dir_id_list[dir_target[0]], img2).replace('\\', '/')
            save_line = '1 {} {}\n'.format(save_dir1, save_dir2)
            all_save_line.append(save_line)

    print('save_data_num : {}'.format(len(all_save_line)))
    save_txt_path = os.path.join(save_txt_path, '{}_pos_{}.txt'.format(image_path.split('/')[-1],len(all_save_line)))
    with open(save_txt_path, 'a', encoding='utf-8') as f:
        f.writelines(all_save_line)

def get_neg_num(image_path,save_txt_path,neg_num):
    save_txt = save_txt_path
    os.makedirs(os.path.dirname(save_txt), exist_ok=True)
    dir_id_list = list(os.listdir(image_path))
    all_save_line = []
    use_img = []
    for i in tqdm(range(neg_num)):
        two_index = random.sample(range(0,len(dir_id_list)), 2)
        dir_id1,dir_id2 = dir_id_list[two_index[0]],dir_id_list[two_index[1]]
        dir1,dir2 = os.path.join(image_path,dir_id1),os.path.join(image_path,dir_id2)
        img_list1,img_list2 = list(os.listdir(dir1)),list(os.listdir(dir2))
        index1,index2 = random.sample(range(0,len(img_list1)),1),random.sample(range(0,len(img_list2)),1)
        img1,img2 = img_list1[index1[0]],img_list2[index2[0]]
        label1,label2 = os.path.join(dir1,img1),os.path.join(dir1,img1)
        #防止出现相同的标签对
        while (label1,label2) in use_img:
            two_index = random.sample(range(0, len(dir_id_list)), 2)
            dir_id1, dir_id2 = dir_id_list[two_index[0]], dir_id_list[two_index[1]]
            dir1, dir2 = os.path.join(image_path, dir_id1), os.path.join(image_path, dir_id2)
            img_list1, img_list2 = list(os.listdir(dir1)), list(os.listdir(dir2))
            index1, index2 = random.sample(range(0, len(img_list1)), 1), random.sample(range(0, len(img_list2)), 1)
            img1, img2 = img_list1[index1[0]], img_list2[index2[0]]
            label1, label2 = os.path.join(dir1, img1), os.path.join(dir1, img1)
        #反过来是一样的
        use_img.append((label1,label2))
        use_img.append((label2,label1))

        if img1.endswith(('.jpg', '.png', '.JPG', '.jpeg')) and img2.endswith(('.jpg', '.png', '.JPG', '.jpeg')):
            save_dir1, save_dir2 = os.path.join(image_path.split('/')[-1],dir_id1, img1).replace('\\','/'), os.path.join(image_path.split('/')[-1],dir_id2, img2).replace('\\','/')
            save_line = '0 {} {}\n'.format(save_dir1,save_dir2)
            all_save_line.append(save_line)

    print('save_data_num : {}'.format(len(all_save_line)))
    save_txt_path = os.path.join(save_txt_path,'{}_neg_{}.txt'.format(image_path.split('/')[-1],neg_num))
    with open(save_txt_path,'a',encoding='utf-8') as f:
        f.writelines(all_save_line)

def get_pos_max(image_path,save_txt_path):
    save_txt = save_txt_path
    os.makedirs(os.path.dirname(save_txt), exist_ok=True)
    dir_id_list = list(os.listdir(image_path))
    all_save_line = []
    for dir_id in tqdm(dir_id_list):
        img_list = list(os.listdir(os.path.join(image_path, dir_id)))
        for i in range(len(img_list)):
            while i < len(img_list)-2:
                img1,img2 = img_list[i],img_list[i+1]
                i += 1
                if img1.endswith(('.jpg', '.png', '.JPG', '.jpeg')) and img2.endswith(('.jpg', '.png', '.JPG', '.jpeg')):
                    save_dir1, save_dir2 = os.path.join(dir_id, img1), os.path.join(dir_id, img2)
                    save_line = '1 {} {}\n'.format(save_dir1, save_dir2)
                    all_save_line.append(save_line)

    print('save_data_num : {}'.format(len(all_save_line)))
    save_txt_path = os.path.join(save_txt_path, 'pos_{}.txt'.format(len(all_save_line)))
    with open(save_txt_path, 'a', encoding='utf-8') as f:
        f.writelines(all_save_line)

def get_val_for_mask(image_path,save_txt_path):
    all_save_line = []
    os.makedirs(os.path.dirname(save_txt_path),exist_ok=True)
    id_list = os.listdir(image_path)
    for id in tqdm(id_list):
        image_dir = os.path.join(image_path,id)
        images = os.listdir(image_dir)
        no_mask = []
        mask = []
        for image in images:
            if image.split(".")[0] in ['1','2']:
                no_mask.append(image)
            else:
                mask.append(image)
        for image1 in no_mask:
            data_1_path = os.path.join(image_dir,image1).replace('/project/share/zyr/data/val/','')
            for image2 in mask:
                data_2_path = os.path.join(image_dir, image2).replace('/project/share/zyr/data/val/', '')
                save_line = '1 {} {}\n'.format(data_1_path,data_2_path)
                all_save_line.append(save_line)

    with open(save_txt_path,'a',encoding='utf-8') as f:
        f.writelines(all_save_line)


def get_data_from_label(image_path,txt_path):
    all_save_line = []
    replace_path = r"D:\using\feature_extraction\val"
    with open(txt_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        label = 0
        for i,line in tqdm(enumerate(lines)):
            line = line.rstrip()
            img1,img2 = line.split(' ')[0],line.split(' ')[-1]
            if img1[0] == '/':
                img1 =img1.replace(img1[0],'',1).replace('/','\\')
            if img2[0] == '/':
                img2 =img2.replace(img2[0],'',1).replace('/','\\')
            img_1,img_2 = os.path.join(image_path,img1),os.path.join(image_path,img2)
            save_name1,save_name2 = '{}_'.format(i)+img_1.split('\\')[-1],'{}_'.format(i)+img_2.split('\\')[-1]
            save_img1,save_img2 = os.path.join(image_path,txt_path.split('\\')[-1].split('.')[0],save_name1),os.path.join(image_path,txt_path.split('\\')[-1].split('.')[0],save_name2)
            # os.makedirs(os.path.dirname(save_img1),exist_ok=True)
            # shutil.copy(img_1,save_img1)
            # shutil.copy(img_2,save_img2)

            if 'neg' in txt_path:
                save_line = '0 {} {}\n'.format(img_1.replace(replace_path+'\\',''),img_2.replace(replace_path+'\\',''))
                save_line = save_line.replace('\\','/')
            else:
                save_line = '1 {} {}\n'.format(img_1.replace(replace_path+"\\",''),img_2.replace(replace_path+'\\',''))
                save_line = save_line.replace('\\', '/')
                label = 1
            all_save_line.append(save_line)
    if label == 0:
        save_txt_path = os.path.join(os.path.join(os.path.dirname(txt_path),'neg_anno.txt'))
    else:
        save_txt_path = os.path.join(os.path.join(os.path.dirname(txt_path), 'pos_anno.txt'))
    with open(save_txt_path,'a',encoding='utf-8') as f:
        f.writelines(all_save_line)

def combined_label(label_path):
    combined_label_path = label_path[0].replace(os.path.basename(label_path[0]),'combined.txt')
    with open(combined_label_path, 'a+', encoding='utf-8') as f:
        for label in tqdm(label_path):
            with open(label,'r',encoding='utf-8') as f1:
                lines = f1.readlines()
                f.writelines(lines)
                f1.close()
    f.close()
    with open(combined_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        combined_num = len(lines)
    f.close()
    save_label_name = combined_label_path.replace('combined.txt','val_merge_combined_{}.txt'.format(combined_num))
    os.rename(combined_label_path,save_label_name)

def main1():
    image_path = r'/project/SphereFace2Data/val_merge'
    save_txt_path = r"/project/SphereFace2Data"
    get_pos_num(image_path, save_txt_path, 5000)
    get_neg_num(image_path, save_txt_path, 5000)

def main2():
    image_path = r'D:\using\feature_extraction\val\1800'
    txt_path = r"D:\using\feature_extraction\val\fusion\positive_pairs.txt"
    get_data_from_label(image_path,txt_path)

def main3():
    label_path = ['/project/SphereFace2Data/val_merge_neg_5000.txt','/project/SphereFace2Data/val_merge_pos_5000.txt']
    combined_label(label_path)

if __name__ == '__main__':
    # main1()
    main3()













