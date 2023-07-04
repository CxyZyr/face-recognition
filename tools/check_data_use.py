from __future__ import print_function
import os
import time
import shutil
import cv2

from tqdm import tqdm
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def get_last_line(txt_path):
    file_name = txt_path
    # 判断文件是否已有数据
    if not os.path.getsize(file_name):
        raise Exception('文件没有内容！')
    with open(file_name, 'rb') as f:
        first_line = f.readline()# 读第一行
        all_line = f.readlines()
        off = -50     # 设置偏移量，偏移量不能太大，如果太大会报错
        while True:
            f.seek(off, 2)  # seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
            lines = f.readlines()  # 读取文件指针范围内所有行
            if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                last_line = lines[-1]  # 取最后一行
                break
            # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
            # 所以off翻倍重新运行，直到readlines不止一行
            off *= 2
        first_line = first_line.decode('utf8')
        last_line = last_line.decode('utf8')
        print('文件 ' + file_name + '第一行为：' + first_line)
        print('文件 ' + file_name + '最后一行为：' + last_line)
        print('文件 ' + file_name + '数据量为：' + str(len(all_line)+1))

def get_max_and_classes(anno_path):
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        max_num = 0
        data_items = []

        for line in tqdm(lines):
            path, name = line.rstrip().split()
            data_items.append({'name': name})
            max_num = max(max_num, int(name))
    names = {item['name'] for item in data_items}
    print('max_iter is {}'.format(max_num))
    print('classes num is {}'.format(len(names)))


def check_dataset_time(anno_path):
    start_time = time.time()
    data_items = []
    label_items = []
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        classes = lines[-1].rstrip().split()[-1]
    for line in tqdm(lines):
        path, label = line.rstrip().split()
        data_items.append(path)
        label_items.append(int(label))
    if len(data_items) == 0:
        raise (RuntimeError('Found 0 files.'))
    end_time = time.time()
    use_time = end_time - start_time
    print('-----classes num is {}'.format(int(classes)+1))
    print('-----Load data use {}s'.format(int(use_time)))
    print('-----data_items size is {:.2f}G'.format(total_size(data_items) / (1024 ** 3)))
    print('-----label_items size is {:.2f}G'.format(total_size(label_items) / (1024 ** 3)))

def check_repeat_class(img_dir):
    list1 = os.listdir(img_dir)
    all_classes = []
    for dir in tqdm(list1):
        if '.zip' in dir:
            continue
        image_dir = os.listdir(os.path.join(img_dir,dir))
        for id in image_dir:
            if id not in all_classes:
                all_classes.append(id)
            else:
                print('--repeat--')



if __name__ == '__main__':
    # path = r'/project/SphereFace2Data/real_train_sphereface2_300w.txt'
    # get_last_line(path)
    # get_max_and_classes(path)
    # check_dataset_time(path)
    path = r'/project/SphereFace2Data/real_train_sphereface2_20230630.txt'
    get_last_line(path)
    get_max_and_classes(path)
    check_dataset_time(path)
