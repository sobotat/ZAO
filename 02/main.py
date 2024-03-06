import json
from os import remove
import cv2 as cv
from pynput import mouse
from locator import Locator, UnableToLocateException

minMatchPercent = 95
windowScaling = 1.5
showDebug = False

def loadAndSaveSettings():
    global windowScaling, showDebug, minMatchPercent

    if input("Edit Config: ").lower() in ["yes", "y"]:
        windowScaling = float(input("Window Scale: "))
        showDebug = input("Show Debug: ").lower() in ["y", "yes", "true"]
        minMatchPercent = int(input("Minimal Match Percent: "))
        with open("setting.json", "w") as file:
            json.dump({
                "win_scale": windowScaling,
                "show_debug": showDebug,
                "min_match_percent": minMatchPercent
            }, file, indent=4)
    else:
        try:
            with open("setting.json", "r") as file:
                data = json.load(file)
                windowScaling = float(data["win_scale"])
                showDebug = bool(data["show_debug"])
                minMatchPercent = int(data["min_match_percent"])
        except FileNotFoundError:
            print("Using Defaults Settings")
        except KeyError:
            remove("setting.json")

if __name__ == "__main__":
    mouseController = mouse.Controller()

    loadAndSaveSettings()
    usePrediction = input("Use Prediction: ").lower() in ["yes", "y", "true"]

    filenames = ["templates/target2.png"]
    maxMovement = 13

    # filenames = ["templates/duck1.png", "templates/duck2.png", "templates/duck3.png", "templates/duck4.png"]
    # maxMovement = 100

    templates = []
    for filename in filenames:
        template = cv.imread(filename)
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        templates.append(template)

    locator = Locator(templates, showDebug, minMatchPercent, maxMovement)

    cv.waitKey(1000)
    lastMousePos = mouseController.position

    try:
        while True:
            if abs(lastMousePos[0] - mouseController.position[0] >= 25) or abs(
                    lastMousePos[1] - mouseController.position[1] >= 25):
                print("\033[1;31mPaused because of mouse movement\033[0m")
                lastMousePos = mouseController.position
                cv.waitKey(4000)
                continue

            print("\033[1;37m--------------------------------\033[0m")

            try:
                center, diff = locator.locate(windowScaling, usePrediction=usePrediction)
                mouseController.position = center
                mouseController.press(mouse.Button.left)
                mouseController.release(mouse.Button.left)

                lastMousePos = center
            except UnableToLocateException as ue:
                print(ue)
            cv.waitKey(650)
    except KeyboardInterrupt:
        print("Exiting")
