import os

import cv2
import random
import numpy as np

class RandomLight(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            """应用光照叠加"""
            # 随机生成alpha和beta值
            alpha = np.random.uniform(0.5, 1.5)
            beta = np.random.uniform(-0.5, 0.5)

            # 应用亮度和对比度增强
            augmented_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha)

            # 将原始图像和增强后的图像进行混合
            img = cv2.addWeighted(img, 1 - 0.5, augmented_image, 0.5, 0)

        return img

class RandomBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用模糊增强"""
            kernel = np.ones((3,3), np.float32) / (3 ** 2)

            # 应用卷积操作
            img = cv2.filter2D(img, -1, kernel)

        return img

class RandomMask(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机遮挡"""
            # 随机生成遮挡区域的位置和大小
            height, width = img.shape[:2]
            mask_width = np.random.randint(1, 64)
            mask_height = np.random.randint(1, 64)
            x1 = np.random.randint(0, width - mask_width)
            y1 = np.random.randint(0, height - mask_height)
            x2 = x1 + mask_width
            y2 = y1 + mask_height

            # 创建遮挡区域并应用于图像
            masked_image = img.copy()
            masked_image[y1:y2, x1:x2] = (0, 0, 0)
            img = masked_image

        return img

class RandomRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机旋转"""
            # 随机选择旋转角度
            angle = np.random.uniform(-15, 15)

            # 获取图像尺寸
            height, width = img.shape[:2]
            center_x, center_y = width / 2, height / 2

            # 计算旋转矩阵并应用于图像
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT_101)

        return img

if __name__ == '__main__':
    img_dir = r'C:\Users\Glassxsix-zhou\Desktop\test'
    imgs_name = os.listdir(img_dir)
    trans = [
        RandomLight(p=0.5),
        RandomBlur(p=0.5),
        RandomMask(p=0.5),
        RandomRotation(p=0.5)
    ]
    for name in imgs_name:
        img_path = os.path.join(img_dir,name)
        img = cv2.imread(img_path)
        blended_image = trans[0](img)
        print(blended_image.shape)
        blurred_image = trans[1](img)
        masked_image = trans[2](img)
        rotated_image = trans[3](img)
        cv2.imshow('origin_image',img)
        # cv2.imshow('blended_image',blended_image)
        # cv2.imshow('blurred_image', blurred_image)
        cv2.imshow('masked_image', masked_image)
        # cv2.imshow('rotated_image', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





