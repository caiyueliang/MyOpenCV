# encoding:utf-8
'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

from numpy import *
import numpy as np
import cv2
import sys
import datetime
# from common import anorm2, draw_str
# from time import clock


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):                                      # 构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 5                                        # 每隔多少帧检测一次特征点
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):                                                      # 光流运行方法
        while True:
            ret, frame = self.cam.read()                                # 读取视频帧
            if ret is True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 转化为灰度虚图像
                vis = frame.copy()

                if len(self.tracks) > 0:                                # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                    # ==========================================================================
                    # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)            # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 4                                        # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks

                    # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255))
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if self.frame_idx % self.detect_interval == 0:              # 每n帧检测一次特征点
                    mask = np.zeros_like(frame_gray)                        # 初始化和视频大小相同的图像
                    mask[:] = 255                                           # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:   # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])                    # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    # 背景减除
    def background_subtractor_MOG2(self):
        cap = self.cam

        fgbg = cv2.createBackgroundSubtractorMOG2()
        # fgbg = cv2.createBackgroundSubtractorKNN()

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # frame1 = np.zeros((640, 480))
        # out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 5.0, np.shape(frame1))

        while (True):
            ret, frame = cap.read()
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)        # 高斯模糊

            fgmask = fgbg.apply(frame)
            cv2.imshow('fgmask', fgmask)
            (_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            maxArea = 0
            for c in cnts:
                Area = cv2.contourArea(c)
                if Area < maxArea:
                    # if cv2.contourArea(c) < 500:
                    (x, y, w, h) = (0, 0, 0, 0)
                    continue
                else:
                    if Area < 1000:
                        (x, y, w, h) = (0, 0, 0, 0)
                        continue
                    else:
                        maxArea = Area
                        m = c
                        (x, y, w, h) = cv2.boundingRect(m)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # out.write(frame)
            cv2.imshow('frame', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        # out.release()
        cap.release()

def main():
    try:
        video_src = sys.argv[1]
    except:
        video_src = "/cyl_data/2018070718.mp4"

    print __doc__
    # App(video_src).run()
    App(video_src).background_subtractor_MOG2()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     cv2.calcOpticalFlowPyrLK()
#     cv2.calcOpticalFlowFarneback()
