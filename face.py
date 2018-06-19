# encoding: utf-8
import cv2
import numpy as np
import time


def get_fullbody(image):
    cvo = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_fullbody.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return image


def get_upperbody(image):
    cvo = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_upperbody.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return image


def get_mouth(image):
    cvo = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_mcs_mouth.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

    return image


def get_nose(image):
    cvo = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_mcs_nose.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return image


def get_eyes(image):
    cvo = cv2.CascadeClassifier('haarcascade_eye.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image


def get_face(image):
    cvo = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


def get_face_1(image):
    cvo = cv2.CascadeClassifier('haarcascade_profileface.xml')
    cvo.load('C:\\Soft\\PythonWorkspace\\MyOpenCV\\opencv_data\\haarcascade_profileface.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别面部
    faces = cvo.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    # 给识别的脸花方框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


# 加载播放video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


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
            frame = get_face(frame)
            frame = get_eyes(frame)
            # frame = get_mouth(frame)
            # frame = get_nose(frame)
            frame = get_fullbody(frame)
            frame = get_upperbody(frame)
            out.write(frame)

            cv2.imshow('My Camera', frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    out.release()
    cam.release()
    cv2.destroyAllWindow()


def get_face_with_image(face_path, save_name):
    # 使用函数cv.imread（）来读取图像。图像应该在工作目录中，或者应该给出图像的完整路径。
    # 第二个参数是一个标志，用于指定应读取图像的方式。
    # cv.IMREAD_COLOR：加载彩色图像。图像的任何透明度都将被忽略。这是默认标志。
    # cv.IMREAD_GRAYSCALE：以灰度模式加载图像
    # cv.IMREAD_UNCHANGED：加载包含Alpha通道的图像
    img = cv2.imread(face_path, cv2.IMREAD_COLOR)

    img = get_face(img)
    # img = get_face_1(img)
    # img = get_eyes(img)
    # img = get_mouth(img)
    # img = get_nose(img)
    img = get_fullbody(img)
    img = get_upperbody(img)

    cv2.imwrite(save_name, img)     # 保存图片

    cv2.imshow('imshow', img)       # 显示图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # get_face_with_video()
    load_video('output.mp4')

    # get_face_with_image('C:\\Soft\\PythonWorkspace\\MyOpenCV\\image\\timg.jpg', 'save.jpg')
    # get_face_with_image('C:\\Soft\\PythonWorkspace\\MyOpenCV\\image\\timg1.jpg', 'save1.jpg')
    # get_face_with_image('C:\\Soft\\PythonWorkspace\\MyOpenCV\\image\\timg2.jpg', 'save2.jpg')
