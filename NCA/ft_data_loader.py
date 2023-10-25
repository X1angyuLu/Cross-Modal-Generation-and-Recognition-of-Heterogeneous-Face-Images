import numpy as np
import os
from os import listdir
from os.path import join, exists, isdir
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABCMeta,abstractmethod
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
import imageio
import cv2
import cfg
import base_cyclegan

g_AB, g_BA = base_cyclegan.get_syn_model()

def GetLists(domain='A'):
    if domain == 'A':
        dir = open(cfg.photo_label_path)
    else:
        dir = open(cfg.sketch_label_path)
    lines = dir.readlines()
    lists = []
    for line in lines:
        lists.append(line.split())
    return lists

def GetTriplet(domain='A', begin=0, end=1194):
    anchor_num = np.random.choice(end-begin)
    anchor_num = anchor_num + begin
    while True:
        negative_num = np.random.choice(end-begin)
        negative_num = negative_num + begin
        if negative_num != anchor_num:
            break
    photo_label = GetLists('A')
    sketch_label = GetLists('B')
    if domain == "A":  # photo matching method
        # img_anchor
        img = imageio.imread(cfg.sketch_path + str(sketch_label[anchor_num][0])).astype(np.float)
        imgs = []
        imgs.append(img)
        imgs = np.array(imgs) / 127.5 - 1.
        img = g_BA.predict(imgs)[0]
        img = 127.5 * img + 127.5
        img_anchor = cv2.resize(img, (224, 224))

        #img_pos
        img_pos = image.load_img(cfg.photo_path + str(photo_label[anchor_num][0]), target_size=(224, 224))
        img_pos = image.img_to_array(img_pos)

        #img_neg
        img_neg = image.load_img(cfg.photo_path + str(photo_label[negative_num][0]), target_size=(224, 224))
        img_neg = image.img_to_array(img_neg)


    else:
        # img_anchor
        img_anchor = image.load_img(cfg.sketch_path + str(sketch_label[anchor_num][0]), target_size=(224, 224))
        img_anchor = image.img_to_array(img_anchor)

        # img_pos
        img = imageio.imread(cfg.photo_path + str(photo_label[anchor_num][0])).astype(np.float)
        imgs = []
        imgs.append(img)
        imgs = np.array(imgs) / 127.5 - 1.
        img = g_AB.predict(imgs)[0]
        img = 127.5 * img + 127.5
        img_pos = cv2.resize(img, (224, 224))

        # img_neg
        img = imageio.imread(cfg.photo_path + str(photo_label[negative_num][0])).astype(np.float)
        imgs = []
        imgs.append(img)
        imgs = np.array(imgs) / 127.5 - 1.
        img = g_AB.predict(imgs)[0]
        img = 127.5 * img + 127.5
        img_neg = cv2.resize(img, (224, 224))


    return img_anchor, img_pos, img_neg






def TripletGenerator(domain='A',begin=0,end=1194):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(1):
            img_anchor, img_pos, img_neg = GetTriplet(domain, begin, end)
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None

        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label)