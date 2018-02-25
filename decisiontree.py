#! /usr/bin/python
# -*- coding: utf-8 -*-
import gdal
import math
from BinaryTree import BinaryTree
import numpy as np
from PIL import Image, ImageDraw


def getdata(filepath):
    data = gdal.Open(filepath)
    w = data.RasterXSize
    h = data.RasterYSize
    r = data.GetRasterBand(1).ReadAsArray(0, 0, w, h).reshape(h * w)
    g = data.GetRasterBand(2).ReadAsArray(0, 0, w, h).reshape(h * w)
    b = data.GetRasterBand(3).ReadAsArray(0, 0, w, h).reshape(h * w)
    img_data = [[r[idx], g[idx], b[idx]] for idx, _ in enumerate(r)]
    img_data = np.array(img_data).reshape((h, w, 3))
    pre_data = img_data.reshape(img_data.shape[0] * img_data.shape[1], img_data.shape[2])
    return img_data, pre_data


def gen_table(img_data):
    tree = img_data[32:115, 401:511]
    house = img_data[338:442, 23:74]
    sea = img_data[518:653, 661:820]
    newtree = np.c_[tree.reshape((tree.shape[0] * tree.shape[1], tree.shape[2])),
                    np.ones(tree.shape[0] * tree.shape[1], dtype='uint8')]
    newhouse = np.c_[house.reshape((house.shape[0] * house.shape[1], house.shape[2])),
                     np.ones(house.shape[0] * house.shape[1], dtype='uint8') * 2]
    newsea = np.c_[sea.reshape((sea.shape[0] * sea.shape[1], sea.shape[2])),
                   np.ones(sea.shape[0] * sea.shape[1], dtype='uint8') * 3]
    train_data = list(np.vstack((newtree, newhouse, newsea)))
    return train_data


def filter_data(data: list):
    data = np.array(data)
    record = dict()
    for d in data:
        if tuple(d[:3]) in record:
            if d[3] in record[tuple(d[:3])]:
                record[tuple(d[:3])][d[3]] += 1
            else:
                record[tuple(d[:3])][d[3]] = 1
        else:
            record[tuple(d[:3])] = {d[3]: 1}
    for idx, d in enumerate(data):
        data[idx][3] = sorted([[k, record[tuple(d[:3])][k]] for k in record[tuple(d[:3])]],
                              key=lambda x: x[1], reverse=True)[0][0]
    return data


def show_img(img_data):
    img = Image.fromarray(img_data, 'RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle([402, 33, 511, 115], fill=None, outline=(43, 255, 41))
    draw.rectangle([24, 339, 74, 442], fill=None, outline=(0, 25, 255))
    draw.rectangle([662, 519, 820, 653], fill=None, outline=(255, 0, 0))
    del draw
    img.show()


def calc_ent(x: int, y: int) -> float:
    s = x + y
    if x == 0 or y == 0:
        return 0
    return -(x / s * math.log2(x / s) + y / s * math.log2(y / s))


def calc_i(data, label):
    x = 0
    y = 0
    for item in data:
        if item[3] == label:
            x += 1
        else:
            y += 1
    return calc_ent(x, y)


def calc_gain(data, column, value, ents, labels, label):
    l = 0
    r = m = len(data)
    lx, ly = 0, 0
    for item in data:
        if item[column] < value:
            l += 1
            r -= 1
            if item[3] == label:
                lx += 1
            else:
                ly += 1
    rx = labels[label] - lx
    ry = m - labels[label] - ly
    sum_ent = l / m * calc_ent(lx, ly) + r / m * calc_ent(rx, ry)
    return ents[label] - sum_ent


def calc_split_i(data, column, value):
    l = 0
    for item in data:
        if item[column] < value:
            l += 1
    return calc_ent(l, len(data) - l)


def calc_gain_radio(data, column, value, ents, labels, label):
    gain = calc_gain(data, column, value, ents, labels, label)
    split = calc_split_i(data, column, value)
    return gain / split


def check_stop_split(data: list):
    data = np.array(data)
    if len(data[data[:, 3] == 1]) == len(data) or len(data[data[:, 3] == 2]) == len(data) or \
       len(data[data[:, 3] == 3]) == len(data):
        return True
    return False


def prune(data: list):
    data = np.array(data)
    record = dict()
    for d in data:
        if d[3] in record:
                record[d[3]] += 1
        else:
            record[d[3]] = 1
    label = sorted([[k, record[k]] for k in record], key=lambda x: x[1])[0][0]
    return label


def get_best_split_node(data: list):
    if len(data) == 1:
        return [0, 1, data[0][3], data[0][0], 0]
    labels = dict()
    ents = dict()
    for d in data:
        if d[3] in labels:
            labels[d[3]] += 1
        else:
            labels[d[3]] = 1
            ents[d[3]] = calc_i(data, d[3])
    datas = list()
    for i in range(3):
        datas.append(sorted(data, key=lambda x: x[i]))
    gain_radios = [-10.0, 0, 0, 0, 0]
    for column in range(3):
        for label in labels.keys():
            for idx in range(1, len(data)):
                if datas[column][idx][column] != datas[column][idx - 1][column]:
                    value = datas[column][idx][column] / 2 + datas[column][idx - 1][column] / 2
                    if gain_radios[0] < calc_gain_radio(datas[column], column, value, ents, labels, label):
                        gain_radios[0] = calc_gain_radio(datas[column], column, value, ents, labels, label)
                        gain_radios[1] = idx
                        gain_radios[2] = label
                        gain_radios[3] = value
                        gain_radios[4] = column
    return gain_radios, datas[gain_radios[4]]


def gen_tree(data: list):
    best_split_node, new_data = get_best_split_node(data)
    if best_split_node == [-10.0, 0, 0, 0, 0]:
        return
    node = BinaryTree((best_split_node[4], best_split_node[3]))
    if not check_stop_split(new_data[:best_split_node[1]]):
        if len(new_data[:best_split_node[1]]) <= 100:
            node.leftChild = BinaryTree(prune(new_data[:best_split_node[1]]))
        else:
            node.leftChild = gen_tree(new_data[:best_split_node[1]])
    else:
        node.leftChild = BinaryTree(best_split_node[2])
    if not check_stop_split(new_data[best_split_node[1]:]):
        if len(new_data[best_split_node[1]:]) <= 100:
            node.rightChild = BinaryTree(prune(new_data[best_split_node[1]:]))
        else:
            node.rightChild = gen_tree(new_data[best_split_node[1]:])
    else:
        node.rightChild = BinaryTree(best_split_node[2])
    return node


def main():
    img_data, pre_data = getdata('HJ1A-CCD2-450-72-20091015.img')
    data = gen_table(img_data)
    data = filter_data(data)
    root = gen_tree(list(data))
    pre_img = np.zeros_like(pre_data)
    for idx, v in enumerate(pre_data):
        if root.predict(v) == 1:
            pre_img[idx] = [0, 255, 0]
        elif root.predict(v) == 2:
            pre_img[idx] = [0, 0, 255]
        else:
            pre_img[idx] = [255, 0, 0]
    pre_img = pre_img.reshape((813, 1440, 3))
    for x in range(1, pre_img.shape[0] - 1):
        for y in range(1, pre_img.shape[1] - 1):
            tmp = [0, 0, 0]
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    tmp[np.argsort(pre_img[i, j])[2]] += 1
            rgb = np.argsort(tmp)[2]
            if rgb == 1:
                pre_img[x, y] = [0, 255, 0]
            elif rgb == 2:
                pre_img[x, y] = [0, 0, 255]
            else:
                pre_img[x, y] = [255, 0, 0]
    img = Image.fromarray(pre_img, 'RGB')
    img.show()

if __name__ == '__main__':
    main()
