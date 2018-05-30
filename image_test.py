import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def object_tracking():
    cap = cv.VideoCapture(0)

    while(True):
        # 拍摄每一帧
        _, frame = cap.read()

        # 将BGR转换为HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 定义HSV中的蓝色范围
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        # 将HSV图像阈值限制为仅获得蓝色
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        # 按位与屏蔽和原始图像
        res = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('frame', frame)
        cv.imshow('hsv', hsv)
        cv.imshow('mask', mask)
        cv.imshow('res', res)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()


def show_image():
    img = cv.imread('image/timg2.jpg')
    cv.imshow('image', img)
    cv.waitKey(0)


# 如何获得某个颜色的HSV值（如绿色）
def get_color_HSV(color):
    hsv_color = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    print(hsv_color)


# 缩放
def scaling_image():
    # 优选的内插方法是cv.INTER_AREA用于收缩和cv.INTER_CUBIC（慢）cv.INTER_LINEAR用于变焦。默认情况下，
    # 对于所有调整大小的目的，使用的插值方法是cv.INTER_LINEAR。您可以使用以下方法调整输入图像大小：
    img = cv.imread('image/timg2.jpg')
    cv.imshow('img', img)
    # res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    # OR
    height, width = img.shape[:2]
    res = cv.resize(img, (1*width, 2*height), interpolation=cv.INTER_CUBIC)
    cv.imshow('scaling', res)
    cv.waitKey(0)
    cv.destroyAllWindows()


# cv.warpAffine（）函数的第三个参数是输出图像的大小，它应该是（width，height）的形式。请记住width =列数，height =行数。
# 翻转（转置矩阵？）
def translation():
    img = cv.imread('image/timg2.jpg', 0)
    cv.imshow('img', img)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))       # 将仿射变换应用于图像
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 旋转图片
def rotation():
    img = cv.imread('image/timg2.jpg', 0)
    cv.imshow('img', img)
    rows, cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 仿射变换
def affine_transformation():
    img = cv.imread('image/timg2.jpg')
    cv.imshow('img', img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 透视变换
def perspective_transformation():
    img = cv.imread('image/timg2.jpg')
    # cv.imshow('img', img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    # cv.imshow('dst', dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


# 边缘检测
def edge_detection():
    img = cv.imread('image/timg2.jpg', 0)
    # edges = cv.Canny(img, 100, 200)
    edges = cv.Canny(img, 100, 150)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # object_tracking()

    # green = np.uint8([[[0, 255, 0]]])
    # get_color_HSV(green)
    # red = np.uint8([[[255, 0, 0]]])
    # get_color_HSV(red)
    # blue = np.uint8([[[0, 0, 255]]])
    # get_color_HSV(blue)

    # 缩放图片
    # scaling_image()

    # 翻转（转置矩阵？）
    # translation()

    # 旋转图片
    # rotation()

    # 仿射变换
    # affine_transformation()

    # 透视变换
    # perspective_transformation()

    # 边缘检测
    edge_detection()
