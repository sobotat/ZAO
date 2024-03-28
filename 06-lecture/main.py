import cv2 as cv
import numpy as np

img1 = cv.imread("template.jpg", 0)
img2 = cv.imread("template_car.jpg", 0)

img1 = cv.resize(img1, (80,80))
img2 = cv.resize(img2, (80,80))

faceRecognizer = cv.face.LBPHFaceRecognizer.create()

if not input("Skip Training: ").lower() in ["y", "yes"]:
    print("\033[1;36mTraining\033[0m")
    trainList = [img1, img2]
    trainLabel = [0, 1]

    print("GridX", faceRecognizer.getGridX())
    print("GridY", faceRecognizer.getGridY())
    print("Neighbors", faceRecognizer.getNeighbors())
    print("Radius", faceRecognizer.getRadius())

    faceRecognizer.train(trainList, np.array(trainLabel))
    faceRecognizer.write("out.yaml")
    print()
else:
    faceRecognizer.read("out.yaml")

print("\033[1;36mPredicting\033[0m")
img3 = cv.imread("template_sun.jpg", 0)
img3 = cv.resize(img3, (80,80))

print("Img1 Empty    ", faceRecognizer.predict(img1))
print("Img2 Car      ", faceRecognizer.predict(img2))
print("Img3 Empty Sun", faceRecognizer.predict(img3))
