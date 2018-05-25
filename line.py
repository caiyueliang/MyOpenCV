import numpy as np
import cv2 as cv


def image_show(img):
    # while (True):
    cv.imshow('[draw]press "q" to exit', img)
    cv.waitKey(0)
    #    if cv.waitKey(1) & 0xFF == ord('q'):
    #        break
    cv.destroyAllWindows()


def draw_line():
    # 创建一个黑色的图像
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    # 绘制一条厚度为5px的对角蓝线
    img = cv.line(img=img, pt1=(0, 0), pt2=(511, 511), color=(255, 0, 0), thickness=5)
    image_show(img)


def draw_rectangle():
    # 创建一个黑色的图像
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    img = cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    image_show(img)


def draw_circle():
    # 创建一个黑色的图像
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    img = cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
    image_show(img)


def draw_ellipse():
    # 创建一个黑色的图像
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    img = cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    image_show(img)


def draw_polylines(close=True):
    # 创建一个黑色的图像
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv.polylines(img, [pts], close, (0, 255, 255))
    font = cv.FONT_HERSHEY_SIMPLEX
    img = cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    image_show(img)


if __name__ == '__main__':
    draw_line()
    draw_rectangle()
    draw_circle()
    draw_ellipse()          # 椭圆
    draw_polylines()        # 多边形（封闭）
    draw_polylines(False)   # 多边形（不封闭）
