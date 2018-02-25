#! /usr/bin/python
# -*- encoding:utf-8 -*-
import numpy
from queue import Queue


class BinaryTree:
    def __init__(self, root_val):
        self.v = root_val
        self.leftChild = None
        self.rightChild = None

    def predict(self, pixel):
        tmp = self
        while tmp is not None:
            # print(tmp.v)
            if isinstance(tmp.v, numpy.uint8):
                return tmp.v
            elif isinstance(tmp.v, tuple):
                if pixel[tmp.v[0]] <= tmp.v[1]:
                    tmp = tmp.leftChild
                elif pixel[tmp.v[0]] > tmp.v[1]:
                    tmp = tmp.rightChild

    def pre_order(self):
        print(self.v, end=' ')
        if self.leftChild is not None:
            self.leftChild.pre_order()
        if self.rightChild is not None:
            self.rightChild.pre_order()

    def pos_order(self):
        if self.leftChild is not None:
            self.leftChild.pos_order()
        if self.rightChild is not None:
            self.rightChild.pos_order()
        print(self.v, end=' ')

    def mid_order(self):
        if self.leftChild is not None:
            self.leftChild.mid_order()
        print(self.v, end=' ')
        if self.rightChild is not None:
            self.rightChild.mid_order()

    def level_order(self):
        q = Queue()
        q.put(self)
        while not q.empty():
            node = q.get()
            print(node.v, end=' ')
            if node.leftChild is not None:
                q.put(node.leftChild)
            else:
                print('L', end=' ')
            if node.rightChild is not None:
                q.put(node.rightChild)
            else:
                print('R', end=' ')

    def high(self):
        l_high, r_high = 0, 0
        if self is not None:
            if self.leftChild is not None:
                l_high = self.leftChild.high()
            if self.rightChild is not None:
                r_high = self.rightChild.high()
            return max(l_high, r_high) + 1
