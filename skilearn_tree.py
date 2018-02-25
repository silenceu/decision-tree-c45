#! /usr/bin/python
# -*- coding:utf-8 -*-
import gdal
import numpy as np
from sklearn import tree
from PIL import Image


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
    train_data = np.vstack((newtree, newhouse, newsea))
    return train_data


def gen_test_data(img_data):
    test_data = img_data.reshape(img_data.shape[0] * img_data.shape[1], img_data.shape[2])
    return test_data


def main():
    img_data, pre_data = getdata('HJ1A-CCD2-450-72-20091015.img')
    data = gen_table(img_data)
    decisiontree = tree.DecisionTreeClassifier()
    x = data[:, :3]
    y = data[:, 3]
    decisiontree = decisiontree.fit(x, y)
    results_label = decisiontree.predict(pre_data)
    pre_img = np.zeros_like(pre_data)
    for idx, v in enumerate(results_label):
        if v == 1:
            pre_img[idx] = [0, 255, 0]
        elif v == 2:
            pre_img[idx] = [0, 0, 255]
        else:
            pre_img[idx] = [255, 0, 0]
    pre_data = pre_data.reshape((813, 1440, 3))
    print(pre_data.shape)
    for x in range(1, pre_data.shape[0] - 1):
        for y in range(1, pre_data.shape[1] - 1):
            tmp = [0, 0, 0]
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    tmp[np.argsort(pre_data[i, j])[2]] += 1
            print(tmp)
            rgb = np.argsort(tmp)[2]
            if rgb == 1:
                pre_data[x, y] = [0, 255, 0]
            elif rgb == 2:
                pre_data[x, y] = [0, 0, 255]
            else:
                pre_data[x, y] = [255, 0, 0]
    img = Image.fromarray(pre_data, 'RGB')
    img.show()


if __name__ == '__main__':
    main()
