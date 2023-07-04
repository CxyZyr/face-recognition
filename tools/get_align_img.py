import os
import cv2

import numpy as np
import cv2
import os
import base64
import threading
from tqdm import tqdm
from pyparser import fparser
from PIL import Image as im
import warnings
warnings.filterwarnings('ignore')

def base64torgb(target):
    data_out = base64.b64decode(target)
    output = np.fromstring(data_out,np.uint8)
    couter = 128*128
    img = []
    for i in range(couter):
        img.append(output[i+couter*2])
        img.append(output[i+couter])
        img.append(output[i])
    img = np.asarray(img).reshape((128,128,3))
    # img = np.swapaxes(img,0,2)

    return img

def align_image(path_list,device):

    fparser.init_faceparser(
        license_key="g6psaWNlbnNlX2lk2SQyYzRlMjUwZS0xODVmLTRkNzMtODY2My0yMmU0YjA0YzM3MDGkcG9ydM1A76ZzZXJ2ZXKvMTgyLjE0MC4yNDAuMTIx",
        working_dir=None,
        model_type=1
    )

    params = {
        "event_id": "test_000",
        "end_point": "align",
        # "image": "D:/Silent-Face-Anti-Spoofing/images/sample/image_F2.jpg",
        "detect": {"min_size": 16,
                   "threshold": 0.5,
                   "format": 1,
                   "do_attributing": True,
                   "device": device
                   }
    }

    for file_path in tqdm(path_list):
        image_dir = os.path.join(r'/project/share/zyr/data/val/1800',file_path)
        image_list = list(os.listdir(image_dir))
        if len(image_list) == 0:
            continue
        for image in image_list:
            img_p = os.path.join(image_dir, image)
            img = cv2.imread(img_p)
            if img is None:
                continue
            ret, detect_result = fparser.parser_pipline(params, data=img)
            if ret["status"]["code"] == -1:
                continue
            else:
                pitch = abs(detect_result['facerectwithfaceinfo_list'][0]['attributes']['pitch'])
                yaw = abs(detect_result['facerectwithfaceinfo_list'][0]['attributes']['yaw'])
                if pitch > 15 or yaw > 15:
                    continue
                result = ret["aligned_images"][0]
                img = base64torgb(result)
                save_path = os.path.join(r'/project/share/zyr/data/val/1800_align', file_path,image.split('.')[0] + '.jpg')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img = im.fromarray(img)
                img.save(save_path)

def main():
    image_path = r'/project/share/zyr/data/val/1800'
    length = len(list(os.listdir(image_path)))
    print(length)
    image_dir_list = list(os.listdir(image_path))

    align_image(image_dir_list,0)


if __name__ == '__main__':
    main()


