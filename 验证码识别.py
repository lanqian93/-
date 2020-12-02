import os

import tensorflow as tf
from PIL import Image

import numpy as np
import random

CAPTCHA_LEN = 4

CAPTCHA_HEIGHT = 25

CAPTCHA_WIDTH = 70

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER + UP_CASE


# 图片转为黑白，3维转1维
def convert2gray(img):
    if len(img.shape)>2:
        img = np.mean(img, -1)
    return img


# 验证码向量转为文本
def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]
    return ''.join(text_list)


# 随机生成权重
def weight_variable(shape, w_alpha=0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


# 随机生成偏置项
def bias_variable(shape, b_alpha=0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


# 局部变量线性组合，步长为1，模式‘SAME’代表卷积后图片尺寸不变，即零边距

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# max pooling,取出区域内最大值为代表特征， 2x2pool，图片尺寸变为1/2

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 三层卷积神经网络计算图
def cnn_graph(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])
    # 第一层

    # filter定义为3x3x1， 输出32个特征, 即32个filter

    w_conv1 = weight_variable([3, 3, 1, 32])

    b_conv1 = bias_variable([32])

    # rulu激活函数

    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))

    # 池化

    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合

    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # 第二层
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)

    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层

    w_conv3 = weight_variable([3, 3, 64, 64])

    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # 全连接层

    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height * image_width * 64, 1024])
    b_fc = bias_variable([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height * image_width * 64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)


    # 全连接层(输出层)

    w_out = weight_variable([1024, len(captcha_list) * captcha_len])
    b_out = bias_variable([len(captcha_list) * captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)

    return y_conv


# 单张图片的向量
def load_image(filename, isFlatten=False):
    isExit = os.path.isfile(filename)
    if isExit == False:
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


# 验证码图片转化为文本
def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    x = tf.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.placeholder(tf.float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # todo模型
        saver.restore(sess, '0.99captcha.model-1400')
        predict = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        vector_list = sess.run(predict, feed_dict={x: image_list, keep_prob: 1})
        vector_list = vector_list.tolist()

        text_list = [vec2text(vector) for vector in vector_list]

        return text_list


if __name__ == '__main__':
    image_a = load_image("test.jpg")
    img_array = np.array(image_a)
    image = convert2gray(img_array)
    image = image.flatten() / 255
    pre_text = captcha2text([image])
    print("识别出来的：", pre_text[0])
