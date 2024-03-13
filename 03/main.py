#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def isParkingEmpty(image, template, threshold: float) -> tuple[bool, float]:
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    tmp_out = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)

    match_percents = (1 - min_val) * 100 - 0.1
    return match_percents > threshold, match_percents

def getResultsForImage(filename: str) -> list:
    try:
        with open(filename, "r") as f:
            data = f.read().strip().split("\n")
            return list(map(lambda x: int(x), data))
    except FileNotFoundError:
        return []

def drawView(imageView, coords:list, indexOfParkingPlace, color):
    cv.line(imageView, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 200, 0), 2)
    cv.line(imageView, (int(coords[2]), int(coords[3])), (int(coords[4]), int(coords[5])), (255, 200, 0), 2)
    cv.line(imageView, (int(coords[4]), int(coords[5])), (int(coords[6]), int(coords[7])), (255, 200, 0), 2)
    centerX = int((int(coords[0]) + int(coords[4])) / 2)
    centerY = int((int(coords[1]) + int(coords[5])) / 2)
    cv.circle(imageView, (centerX, centerY), 16, color, -1)
    cv.putText(imageView, str(indexOfParkingPlace), (centerX, centerY + 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

def f1_score(true, predicted):
    TP = FP = FN = 0
    for p, t in zip(predicted, true):
        if p == t:
            TP += 1
        elif p == 1 and t == 0:
            FP += 1
        else:
            FN += 1

    print(TP, FP, FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def main(argv):
    useSlowMode = input("Use slow mode: ").lower() in ["y", "yes", "t", "true"]

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    template = cv.imread("template.jpg", 1)
    templateSun = cv.imread("template_sun.jpg", 1)

    testImages = [img for img in glob.glob("images/*.jpg")]
    testImages.sort()
    testImagesResults = [img for img in glob.glob("images/*.txt")]
    testImagesResults.sort()

    if useSlowMode:
        cv.namedWindow("one_place", cv.WINDOW_NORMAL)
    cv.namedWindow("image", cv.WINDOW_NORMAL)

    for i in range(len(testImages)):
        print(f"\n\033[1;36mImage:\033[0m {testImages[i]}")
        image = cv.imread(testImages[i], 1)
        imageView = image.copy()

        detected = []
        countOfEmpty = 0
        indexOfParkingPlace = 0
        for coords in pkm_coordinates:
            indexOfParkingPlace += 1
            onePlaceImage = four_point_transform(image, coords).copy()
            onePlaceImageRes = cv.resize(onePlaceImage, (80, 80))
            if useSlowMode:
                cv.imshow("one_place", onePlaceImageRes)
                # cv.imwrite("template_sun.jpg", onePlaceImageRes)

            isEmptyBasic, percentsBasic = isParkingEmpty(onePlaceImageRes, template, 91.2)
            isEmptySun, percentsSun = isParkingEmpty(onePlaceImageRes, templateSun, 95.5)

            console = "\033[1;32mEmpty\033[0m" if isEmptyBasic else "\033[1;31mOccupied\033[0m"
            consoleSun = "\033[1;32mEmpty\033[0m" if isEmptySun else "\033[1;31mOccupied\033[0m"
            print(f"[{indexOfParkingPlace}]: {console} [{round(percentsBasic, 2)}%], {consoleSun} [{round(percentsSun, 2)}%]")

            isEmpty = isEmptyBasic or isEmptySun
            color = (50, 255, 0) if isEmpty else (50, 0, 255)
            detected.append(int(not isEmpty))
            if isEmpty:
                countOfEmpty += 1

            drawView(imageView, coords, indexOfParkingPlace, color)

            cv.imshow("image", imageView)
            if useSlowMode:
                if cv.waitKey(500) == ord('q'):
                    exit(0)

        print(f"\033[1;36mNumber of Empty:\033[0m [{countOfEmpty}/{len(pkm_coordinates)}]")
        result = getResultsForImage(testImagesResults[i])
        print(f"\033[1;36mShould be Empty:\033[0m [{len(result) - sum(result)}/{len(pkm_coordinates)}]")
        print(f"\033[1;36mF1 Score:\033[0m {f1_score(result, detected)}")
        if cv.waitKey() == ord('q'):
            exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
