import cv2
import numpy as np


def get_face(image):
    cvo = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)   # 调用计算机摄像头，一般默认为0
    print('cam', cam, cam.isOpened())

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # 定义编码
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))  # 创建videowriter对象

    while(cam.isOpened() is True):
        # 读取帧摄像头
        ret, frame = cam.read()
        if ret is True:
            # 输出当前帧
            frame = get_face(frame)
            print(frame)
            out.write(frame)

            cv2.imshow('My Camera', frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    out.release()
    cam.release()
    cv2.destroyWindow()
