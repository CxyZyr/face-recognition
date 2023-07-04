import os
from tqdm import tqdm

def check_data_and_get_label(img_dir):
    all_save_line1 = []
    all_save_line2 = []
    label = 3994826
    save_txt_path = r'/project/SphereFace2Data/real_train_sphereface2_SocialSecurityAligned.txt'
    save_num_path = r'/project/SphereFace2Data/img_list_SocialSecurityAligned.txt'

    img_dir2 = list(os.listdir(img_dir))
    for list1 in tqdm(img_dir2):
        if '.zip' in list1:
            continue
        img_dir3 = os.path.join(img_dir,list1)
        img_dir4 = list(os.listdir(img_dir3))
        for list2 in img_dir4:
            img_dir5 = os.path.join(img_dir3,list2)
            img_dir6 = list(os.listdir(img_dir5))
            save_line2 = '{} {}\n'.format(img_dir5.replace(r'/project/SphereFace2Data/',''),len(img_dir6))
            all_save_line2.append(save_line2)
            if len(img_dir6) < 1:
                continue
            for img in img_dir6:
                img_path = os.path.join(img_dir5,img)
                save_line1 = '{} {}\n'.format(img_path.replace(r'/project/SphereFace2Data/',''),label)
                all_save_line1.append(save_line1)
            label += 1

    with open(save_txt_path, 'a', encoding='utf-8') as f1:
        f1.writelines(all_save_line1)
    f1.close()
    with open(save_num_path, 'a', encoding='utf-8') as f2:
        f2.writelines(all_save_line2)
    f2.close()

def check_data_and_get_label2(img_dir):
    all_save_line1 = []
    all_save_line2 = []
    label = 3410780
    save_txt_path = r'/project/SphereFace2Data/real_train_sphereface2_cc_data_1_align.txt'
    save_num_path = r'/project/SphereFace2Data/img_list_cc_data_1_align.txt'

    img_dir2 = list(os.listdir(img_dir))
    for list1 in tqdm(img_dir2):
        img_dir3 = os.path.join(img_dir,list1)
        img_dir4 = list(os.listdir(img_dir3))
        save_line2 = '{} {}\n'.format(img_dir3.replace(r'/project/SphereFace2Data/', ''), len(img_dir4))
        all_save_line2.append(save_line2)
        # if len(img_dir4) <= 1:
        #     continue
        for list2 in img_dir4:
            img_path = os.path.join(img_dir3, list2)
            save_line1 = '{} {}\n'.format(img_path.replace(r'/project/SphereFace2Data/', ''), label)
            all_save_line1.append(save_line1)
        label += 1

    with open(save_txt_path, 'a', encoding='utf-8') as f1:
        f1.writelines(all_save_line1)
    f1.close()
    with open(save_num_path, 'a', encoding='utf-8') as f2:
        f2.writelines(all_save_line2)
    f2.close()

def check_data_and_get_label2_sample(img_dir):
    all_save_line1 = []
    label = 0
    num = 0
    save_txt_path = r'/project/SphereFace2Data/real_train_sphereface2_DigiFace1M_align.txt'

    img_dir2 = list(os.listdir(img_dir))
    for list1 in tqdm(img_dir2):
        img_dir3 = os.path.join(img_dir,list1)
        img_dir4 = list(os.listdir(img_dir3))
        # if len(img_dir4) <= 1:
        #     continue
        num2 = 0
        for list2 in img_dir4:
            img_path = os.path.join(img_dir3, list2)
            save_path = img_path.replace(r'/project/SphereFace2Data/', '').replace(list1,str(num),1).replace(list2,str(num2),1)
            save_line1 = '{} {}\n'.format(save_path, label)
            all_save_line1.append(save_line1)
            num2 +=1
        num += 1
        label += 1

    with open(save_txt_path, 'a', encoding='utf-8') as f1:
        f1.writelines(all_save_line1)
    f1.close()

def check_data_and_get_label_sample(img_dir):
    all_save_line1 = []
    label = 606172
    save_txt_path = r'/project/SphereFace2Data/test_GFDExtra.txt'
    num1 = 0
    img_dir2 = list(os.listdir(img_dir))
    for list1 in tqdm(img_dir2):
        num1 += 1
        img_dir3 = os.path.join(img_dir,list1)
        img_dir4 = list(os.listdir(img_dir3))
        num2 = 0
        for list2 in img_dir4:
            img_dir5 = os.path.join(img_dir3,list2)
            img_dir6 = list(os.listdir(img_dir5))
            if len(img_dir6) <= 1:
                continue
            num3 = 0
            for img in img_dir6:
                img_path = os.path.join(img_dir5,img)
                save_path = img_path.replace(r'/project/SphereFace2Data/','').replace(list1,str(num1),1).replace(list2,str(num2),1).replace(img,str(num3),1)
                save_line1 = '{} {}\n'.format(save_path,label)
                all_save_line1.append(save_line1)
                num3 += 1
            label += 1
            num2 += 1

    with open(save_txt_path, 'a', encoding='utf-8') as f1:
        f1.writelines(all_save_line1)
    f1.close()

def combined_txt(label_list, save_txt_path):
    with open(save_txt_path, 'a+', encoding='utf-8') as f:
        for label_txt in tqdm(label_list):
            label_txt_path = os.path.join('/project/SphereFace2Data', label_txt)
            print(label_txt_path)
            with open(label_txt_path, 'r', encoding='utf-8') as f1:
                lines = f1.readlines()
                f.writelines(lines)
                print('-----Load {} is Successful-----'.format(label_txt))

if __name__ == '__main__':
    # img_dir = r'/project/SphereFace2Data/SocialSecurityAligned'
    # check_data_and_get_label(img_dir)
    save_txt_path = r'/project/SphereFace2Data/real_train_sphereface2_20230630.txt'
    label_list = ['real_train_sphereface2_300w.txt','real_train_sphereface2_zjk_align.txt','real_train_sphereface2_cc_data_1_align.txt','real_train_sphereface2_SocialSecurityAligned.txt']
    combined_txt(label_list,save_txt_path)


