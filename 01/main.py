import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")

def cv01():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    img1 = cv2.imread('img.png', 0)
    img2 = cv2.imread('img.png', 0)
    img2 = cv2.medianBlur(img2, 17)

    img_hc = cv2.hconcat([img1, img2])
    cv2.imshow('img', img_hc)
    cv2.waitKey()

def cv02():
    img1 = cv2.imread('img.png', 1)
    img2 = cv2.imread('img.png', 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (200, 200))

    plt.subplot(2, 1, 1)
    plt.imshow(img1)
    plt.subplot(2, 1, 2)
    plt.imshow(img2)
    plt.show()

def cv03():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    img = np.zeros((5, 5, 3), dtype=np.uint8)

    cv2.imshow('img', img)
    cv2.waitKey()

def cv04():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    writer = cv2.VideoWriter('./01/filename.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    while True:
        ret, frame = cap.read()
        edges = cv2.Canny(frame, 50, 100)
        if ret:
            cv2.imshow('img', edges)
            writer.write(frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print('Failed to get Camera')
            break



def main():
    print(cv2.__version__)
    cv04()

if __name__ == '__main__':
    main()
    print('End')