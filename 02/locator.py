
import cv2 as cv
import numpy as np
from numpy import ndarray
from PIL import ImageGrab

class Locator:

    def __init__(self, templates: list[ndarray], showDebug:bool = False, minMatchPercent:int = 95):
        self.templates = templates
        self.boxBorder = 150
        self.showDebug = showDebug
        self.minMatchPercent = minMatchPercent

        if showDebug:
            cv.namedWindow("debug-1", 0)
            cv.namedWindow("debug-2", 0)

    def locate(self, windowScaling:float) -> tuple:

        center_1 = self.captureAndFindOnScreenshot(index=1)
        cv.waitKey(35)
        center_2 = self.captureAndFindOnScreenshot(bbox=(center_1[0] - self.boxBorder, center_1[1] - self.boxBorder, center_1[0] + self.boxBorder, center_1[1] + self.boxBorder), index=2)

        diff = self.getDiff(center_2, center_1)
        diff = diff
        center = center_2

        offset = 0.3
        center = (center[0] + diff[0] * offset, center[1] + diff[1] * offset)
        center = center[0] * 1 / windowScaling, center[1] * 1 / windowScaling

        diffThreshold = 13
        if abs(diff[0]) >= diffThreshold or abs(diff[1]) >= diffThreshold:
            raise UnableToLocateException(f"\033[1;31mLarge Movement:\033[0m Diffs[{round(diff[0], 2)} {round(diff[1], 2)}] >= {diffThreshold}")

        print(f"\033[1;34mLocation:\033[0m [{round(center[0], 2)} {round(center[1], 2)}]")
        print(f"\033[1;34mDifference:\033[0m {diff}")
        return center

    @staticmethod
    def getDiff(a: tuple, b: tuple) -> tuple:
        return a[0] - b[0], a[1] - b[1]

    def captureAndFindOnScreenshot(self, bbox:tuple | None = None, index:int = 1):
        screenshot = self.captureScreenshot(bbox)
        screenshot_debug = screenshot.copy()

        center = None
        for template in self.templates:
            center = self.findOnScreenshot(screenshot, screenshot_debug, template)

            if center is not None:
                break

        if self.showDebug:
            cv.imshow(f"debug-{index}", screenshot_debug)

        if center is None:
            raise UnableToLocateException(f"\033[1;33mNot Found {index}\033[0m")

        if bbox is None:
            return center
        return center[0] + bbox[0], center[0] + bbox[1]

    @staticmethod
    def captureScreenshot(bbox: tuple | None = None) -> ndarray:
        image_grab = ImageGrab.grab(bbox=bbox)
        screen_mat = np.array(image_grab)
        screen_mat = cv.cvtColor(screen_mat, cv.COLOR_BGR2RGB)
        return screen_mat

    def findOnScreenshot(self, screenshot, screenshot_debug, template) -> tuple | None:

        template_h = template.shape[0]
        template_w = template.shape[1]

        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        tmp_out = cv.matchTemplate(screenshot, template, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)

        cv.circle(screenshot_debug, min_loc, 16, (255, 0, 0), cv.FILLED)
        cv.circle(screenshot_debug, max_loc, 16, (0, 255, 0), cv.FILLED)

        match_percents = (1 - min_val) * 100 - 0.1
        print(f"\033[1;36mMatch Percents:\033[0m [{round(match_percents, 2)}%]")
        if match_percents < self.minMatchPercent:
            return None

        top_left = min_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)
        return center_x, center_y

class UnableToLocateException(Exception):
    def __init__(self, message:str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"