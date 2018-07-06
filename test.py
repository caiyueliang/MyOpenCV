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


# ======================================================================================================================
def detectFace(im):
    start = time.time()
    im = cv2.resize(im, (1024, 1024))
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    im_tensor = im_tensor.float().div(255)

    # loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0)))
    # boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0).cpu() if USE_CUDA else loc.data.squeeze(0), F.softmax(conf.squeeze(0), 1).data.cpu() if USE_CUDA else F.softmax(conf.squeeze(0), 1).data)
    # ps = []
    # for p in probs:
    #     pitem = p.item() if torch.is_tensor(p) else p
    #     ps.append(pitem)
    print('detectFace time:', time.time() - start)
    # return boxes, ps


def detectFace_1(im):
    start = time.time()
    im = cv2.resize(im, (1024, 1024))
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    im_tensor = Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0))
    im_tensor = im_tensor.float().div(255)

    # loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0).cuda() if USE_CUDA else torch.unsqueeze(im_tensor, 0)))
    # boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0).cpu() if USE_CUDA else loc.data.squeeze(0), F.softmax(conf.squeeze(0), 1).data.cpu() if USE_CUDA else F.softmax(conf.squeeze(0), 1).data)
    # ps = []
    # for p in probs:
    #     pitem = p.item() if torch.is_tensor(p) else p
    #     ps.append(pitem)
    print('detectFace_1 time:', time.time() - start)
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
            detectFace(frame)
            detectFace_1(frame)
            cv2.imshow('MyCamera', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    out.release()
    cam.release()
    cv2.destroyAllWindow()


def test():
    img = cv2.imread('568640.jpg')
    # img = cv2.resize(img, (800, 480))
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[50, 34], [200, 50], [50, 186]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('Input', img)
    cv2.imshow('Output', dst)
    cv2.imwrite('111.jpg', dst)
    cv2.waitKey(0)
    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
    return

def test_1():
    img = cv2.imread('568640.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))

    cv2.imshow('Input', img)
    cv2.imshow('Output', dst)
    cv2.waitKey(0)
    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
    return

if __name__ == '__main__':
    # get_face_with_video()
    test()
