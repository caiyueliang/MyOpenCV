import cv2
import numpy as np
import time


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


def get_face_with_video():
    cam = cv2.VideoCapture(0)   # 调用计算机摄像头，一般默认为0

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
            frame = get_eyes(frame)
            frame = get_mouth(frame)
            frame = get_nose(frame)
            out.write(frame)

            cv2.imshow('My Camera', frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    out.release()
    cam.release()
    cv2.destroyWindow()


def get_face_with_image(face_path):
    img = cv2.imread(face_path)
    img = get_face(img)
    img = get_face_1(img)
    img = get_eyes(img)
    img = get_mouth(img)
    img = get_nose(img)
    cv2.imwrite('data.jpg', img)


if __name__ == '__main__':
    # get_face_with_video()

    get_face_with_image('C:\\Soft\\PythonWorkspace\\MyOpenCV\\image\\image.jpg')
