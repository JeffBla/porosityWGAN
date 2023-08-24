# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkIOImage import vtkDICOMImageReader

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

shrinkToCenter = 0


def checkInCircle(cx, cy, r, idxX, idxY) -> bool:
    if (idxX - cx)**2 + (idxY - cy)**2 < r**2:
        return True
    else:
        return False


def getTargetCtDataset(inDirname, isStore=False):
    reader = vtkDICOMImageReader()
    reader.SetDirectoryName(inDirname)
    reader.Update()

    files = os.listdir(inDirname)

    dcmImage_CT = np.array(
        reader.GetOutput().GetPointData().GetScalars()).reshape(
            len(files), reader.GetHeight(), reader.GetWidth())

    targetHuList = []
    for index, filename in enumerate(files):
        # 提取像素數據
        # CT value
        Hu = np.flipud(dcmImage_CT[index])

        px_arr = (Hu - reader.GetRescaleOffset()) / reader.GetRescaleSlope()

        # # rescale original 16 bit image to 8 bit values [0,255]
        x0 = np.min(px_arr)
        x1 = np.max(px_arr)
        y0 = 0
        y1 = 255.0
        i8 = ((px_arr - x0) * ((y1 - y0) / (x1 - x0))) + y0

        # # create new array with rescaled values and unsigned 8 bit data type
        o8 = i8.astype(np.uint8)

        # print(f"rescaled data type={o8.dtype}")

        # do the Hough transform
        img = o8
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        imgCanny = cv.Canny(img, 30, 150)

        circles = cv.HoughCircles(imgCanny,
                                  cv.HOUGH_GRADIENT,
                                  2,
                                  20,
                                  param1=70,
                                  param2=90,
                                  minRadius=110,
                                  maxRadius=130)

        # area finding
        # Threshold the image to create a binary image
        ret, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, 2, 1)

        cnt = contours
        big_contour = []
        max = 0
        for i in cnt:
            area = cv.contourArea(
                i)  #--- find the contour having biggest area ---
            if (area > max):
                max = area
                big_contour = i

        # Inside circles
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sometimes, I will many cycles in a image.
            # The way I choose is based on the y vale
            # I choose the topest of the center of circle.
            argmax = np.argmin(circles[0, :, 1])

            circle = circles[0, argmax]

            # calculate porosity
            numOfVoxel = 0
            circleHuList = np.array([])
            for idx, j in np.ndenumerate(Hu):
                # check is inside the circle and coutour?
                # pointPolygonTest -> positive (inside), negative (outside), or zero (on an edge)
                if (checkInCircle(circle[0], circle[1],
                                  circle[2] - shrinkToCenter, idx[1], idx[0])
                        and
                    (cv.pointPolygonTest(big_contour,
                                         (idx[1], idx[0]), False) > 0)):
                    numOfVoxel += 1
                    circleHuList = np.append(circleHuList, Hu[idx[0], idx[1]])

            if circleHuList.size != 0:
                targetHuList.append(circleHuList)
                # output the ct array to file
                if isStore:
                    outputPath = Path('./targetNp', filename) + '.npy'
                    with open(outputPath, 'wb') as f:
                        np.save(f, circleHuList)
            else:
                print('circleHuList is empty.')
    return targetHuList