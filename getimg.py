import os
from PIL import Image
import numpy as np
CAPTCHA_LEN = 4

CAPTCHA_HEIGHT = 25

CAPTCHA_WIDTH = 70

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER + UP_CASE


def file_name(file_dir):
    for files in os.walk(file_dir):
        return files[2]   # 当前路径下所有非目录子文件


def load_image(filename,isFlatten=False):
    isExit=os.path.isfile(filename)
    if isExit==False:
        print("打开失败 ")
    img = Image.open(filename)

    if isFlatten:
        img_flatten = np.array(np.array(img).flatten())
        # print(img_flatten)
        return img_flatten
    else:
        img_arr = np.array(img)
        # print(img_arr)
        return img_arr


def load_allimg():
    flies = file_name("./sample/test")
    list1 = []
    for item in flies:
        # print(item)
        list1.append(load_image("./sample/test/"+str(item)))
    return list1

