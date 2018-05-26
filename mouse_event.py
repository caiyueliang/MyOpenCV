import cv2 as cv
import numpy as np

# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']
events = [i for i in dir(cv) if 'EVENT' in i]
l_button_click_flag = False
r_button_click_flag = False


# mouse callback function
def mouse_click_events(event, x, y, flags, param):
    global l_button_click_flag
    if event == cv.EVENT_LBUTTONDBLCLK:                 # 鼠标左键双击事件响应
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)
    elif event == cv.EVENT_LBUTTONDOWN:
        l_button_click_flag = True
    elif event == cv.EVENT_LBUTTONUP:
        l_button_click_flag = False
    elif event == cv.EVENT_MOUSEMOVE:                     # 鼠标左键长安事件响应
        if l_button_click_flag is True:
            cv.circle(img, (x, y), 3, (255, 255, 0), -1)


drawing = False     # true if mouse is pressed
mode = True         # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1


# mouse callback function
def mouse_click_events_1(event, x, y, flags, param):
    global ix, iy, drawing, mode
    line_width = 2

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing is True:
            if mode is True:
                cv.rectangle(img, (ix, iy), (x - line_width, y - line_width), (0, 0, 0), line_width)
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), line_width)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode is True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), line_width)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        if mode is True:
            mode = False
        else:
            mode = True


if __name__ == '__main__':
    # Create a black image, a window and bind the function to window
    img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    # cv.setMouseCallback('image', mouse_click_events)    # 鼠标事件绑定
    cv.setMouseCallback('image', mouse_click_events_1)  # 鼠标事件绑定
    while (True):
        cv.imshow('image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
