# encoding: utf-8
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from logging import getLogger, DEBUG, basicConfig

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()


def detectFace(im):
    start = time.time()
    im = cv2.resize(im, (1024, 1024))
    # print(im)
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    # print(im_tensor)
    im_tensor = im_tensor.float().div(255)
    # print(im_tensor)

    # loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0)))
    # boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0).cpu() if USE_CUDA else loc.data.squeeze(0), F.softmax(conf.squeeze(0), 1).data.cpu() if USE_CUDA else F.softmax(conf.squeeze(0), 1).data)
    # ps = []
    # for p in probs:
    #     pitem = p.item() if torch.is_tensor(p) else p
    #     ps.append(pitem)
    print('detectFace time:', time.time() - start)
    # print(im_tensor)
    # return boxes, ps


def detectFace_1(im):
    start = time.time()
    im = cv2.resize(im, (1024, 1024))
    # print(im)
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    # print(im_tensor)
    # print('USE_CUDA', USE_CUDA)
    # if USE_CUDA:
    #     im_tensor = Variable(torch.unsqueeze(im_tensor, 0).cuda())
    # else:
    #     im_tensor = Variable(torch.unsqueeze(im_tensor, 0))
    im_tensor = Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0))
    print(1)
    im_tensor = im_tensor.float().div(255)
    # print(im_tensor)

    # loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0)))
    # boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0).cpu() if USE_CUDA else loc.data.squeeze(0), F.softmax(conf.squeeze(0), 1).data.cpu() if USE_CUDA else F.softmax(conf.squeeze(0), 1).data)
    # ps = []
    # for p in probs:
    #     pitem = p.item() if torch.is_tensor(p) else p
    #     ps.append(pitem)
    print('detectFace_1 time:', time.time() - start)
    # print(im_tensor)

    # return boxes, ps


def get_face_with_video():
    cam = cv2.VideoCapture(0)                                           # 调用计算机摄像头，一般默认为0

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # 定义编码
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))  # 创建videowriter对象

    while(cam.isOpened() is True):
        ret, frame = cam.read()             # 逐帧捕获
        if ret is True:
            # 输出当前帧
            # frame = get_face(frame)
            # frame = get_eyes(frame)
            # frame = get_mouth(frame)
            # frame = get_nose(frame)
            # frame = get_fullbody(frame)
            # frame = get_upperbody(frame)
            # out.write(frame)

            cv2.imshow('MyCamera', frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    out.release()
    cam.release()
    cv2.destroyAllWindow()


if __name__ == '__main__':
    get_face_with_video()

    # # data = Variable(torch.randn(1, 3, 1024, 1024))
    # a = np.random.random((1024, 1024, 3))
    # # print('data size', data.size())
    # print('a size', a.shape)
    # detectFace(a)
    # detectFace_1(a)