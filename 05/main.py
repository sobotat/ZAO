import cv2 as cv
import numpy as np

def useCascade(cascade, frame, minNeighbors = 7):
    faces = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=minNeighbors, minSize=(50, 50), maxSize=(500, 500))
    return len(faces) != 0, faces

def remove_duplicates_with_threshold(data, threshold):
    unique_data = []
    for row in data:
        is_duplicate = False
        for existing_row in unique_data:
            if all(abs(a - b) <= threshold for a, b in zip(row, existing_row)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_data.append(row)
    return unique_data

def getCrop(image, coords):
    start_x, start_y, width, height = coords
    end_x = start_x + width
    end_y = start_y + height

    onePlaceImageRes = image[start_y:end_y, start_x:end_x]

    return onePlaceImageRes, start_x, start_y

def isEyeOpened(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    cv.imshow("Preprocessed Image", blurred)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 1, param1=50, param2=20, minRadius=10, maxRadius=40)

    if circles is not None:
        out = []
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            out.append((center,radius))
        return out

    return None

def main():
    cv.namedWindow("win", 0)
    video = cv.VideoCapture("fusek_face_car_01.avi")
    face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    profile_cascade = cv.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    eye_cascade = cv.CascadeClassifier("eye_cascade_fusek.xml")
    smile_cascade = cv.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    template = cv.imread("template.png")

    while True:
        ret, frame = video.read()
        if frame is None:
            break
        paint_frame = frame.copy()
        if ret is True:
            faces = []
            faceDetected, location = useCascade(face_cascade, frame)
            for item in location:
                faces.append(item)
            profileDetected, location = useCascade(profile_cascade, frame)
            for item in location:
                faces.append(item)

            faces = remove_duplicates_with_threshold(faces, 100)
            for face in faces:
                cv.rectangle(paint_frame, face, (0, 255, 0), 6)
                cv.rectangle(paint_frame, face, (255, 255, 255), 2)

                cropped, offsetX, offsetY = getCrop(frame, face)
                eyeDetected, location = useCascade(eye_cascade, cropped, minNeighbors=5)

                # cv.line(paint_frame, [face[0] + face[2] // 2, face[1]], [face[0] + face[2] // 2, face[1] + face[3]], (255,255,255), 2)
                # cv.line(paint_frame, [face[0], face[1] + face[3] // 3 * 2], [face[0] + face[2], face[1] + face[3] // 3 * 2],(255, 255, 255), 2)

                for eye in location:
                    cropped2, offsetX2, offsetY2 = getCrop(frame, [eye[0] + offsetX, eye[1] + offsetY, eye[2], eye[3]])
                    iris = isEyeOpened(cropped2)
                    isOpen = iris is not None

                    if eye[1] + offsetY >= face[1] + face[3] // 3 * 2:
                        continue
                    if eye[0] + offsetX + eye[2] // 2 >= face[0] + face[2] // 2:
                        cv.rectangle(paint_frame, [eye[0] + offsetX, eye[1] + offsetY, eye[2], eye[3]], (0, 0, 255), 2)
                    else:
                        cv.rectangle(paint_frame, [eye[0] + offsetX, eye[1] + offsetY, eye[2], eye[3]], (255, 0, 0), 2)

                    cv.circle(paint_frame, [eye[0] + offsetX, eye[1] + offsetY], 10, (0, 255, 0) if isOpen else (0, 0, 255), cv.FILLED)

                smileDetected, smiles = useCascade(smile_cascade, cropped, minNeighbors=10)
                for smile in smiles:
                    smile = [smile[0] + offsetX, smile[1] + offsetY, smile[2], smile[3]]
                    if smile[1] < face[1] + face[3] // 3 * 2:
                        continue
                    cv.rectangle(paint_frame, smile, (0, 255, 255), 6)
                    cv.rectangle(paint_frame, smile, (255, 255, 255), 2)

        cv.imshow("win", paint_frame)
        if cv.waitKey(2) == ord("q"):
            break

if __name__ == "__main__":
    main()
