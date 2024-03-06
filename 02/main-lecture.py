import cv2 as cv
import numpy as np

A = [4, 5, 8, 2, 9, 9, 1, 3, 7, 4, 5]
B = [6, 7, 9, 3, 9, 3, 1, 6, 7]

SSD = 0
CC = 0
SAD = 0

for i in range(len(A) if len(A) <= len(B) else len(B)):
    SAD += np.abs(A[i] - B[i])
    SSD += (A[i] - B[i]) ** 2
    CC += A[i] * B[i]

print("\033[1;36mInfo:\033[0m")
print(f" SSD[{SSD}]\n CC[{CC}]\n SAD[{SAD}]")

A = np.array(A, dtype=np.uint8)
B = np.array(B, dtype=np.uint8)
TM_SQDIFF = cv.matchTemplate(A, B, cv.TM_SQDIFF)
print("\n\033[1;36mTM_SQDIFF:\033[0m\n", TM_SQDIFF)

# ----------------------------------------------------

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cv.namedWindow("img", 0)
cv.namedWindow("out", 0)

template = cv.imread("templates/image2.png", 0)
cv.imshow("template", template)

while True:
    ret, frame = cap.read()
    if ret:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        tmp_out = cv.matchTemplate(frame_gray, template, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)
        cv.circle(frame, min_loc, 30, (0, 255, 0), -1)
        cv.imshow('img', frame)
        cv.imshow('out', tmp_out)
        if cv.waitKey(1) == ord('q'):
            break
    else:
        print('Failed to get Camera')
        break
