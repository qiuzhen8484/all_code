
import numpy as np
import time
import cv2
from PIL import Image
import os

class ImageData:
    def __init__(self):
        self.path = None
        self.imageData = None
        self.outputData = None
        self.weight = None
        self.height = None
        self.x_left = 0
        self.x_right = 1000
        self.y_begin = 0

    def resetState(self):
        del self.imageData
        self.imageData = None
        self.outputData = None
        self.weight = None
        self.height = None

    def GetIndex(self, ay, thresold, start, length, state):
        index = 0
        if state:
            for i in range(start, length-16, 3):
                # total = (ay[i-7] + ay[i-6] + ay[i-5] + ay[i-4] + ay[i-3] + ay[i-2]
                #     + ay[i-1] + ay[i] + ay[i+1] + ay[i+2] + ay[i+3] + ay[i+4] + ay[i+5] + ay[i+6]
                #     + ay[i+7]) / 15

                # total = (ay[i] + ay[i+1] + ay[i+2] + ay[i+3] + ay[i+4] + ay[i+5] + ay[i+6] + ay[i+7] + ay[i+8] + ay[i+9]
                #          + ay[i+10] + ay[i+11] + ay[i+12] + ay[i+13] + ay[i+14] + ay[i+15]) / 16

                total = (ay[i] + ay[i + 1] + ay[i + 2] + ay[i + 3] + ay[i + 4]) / 5

                if total >= thresold:
                    index = i
                    break
        else:
            for i in range(start, length-16, 1):
                # total = (ay[i] + ay[i+1] + ay[i+2] + ay[i+3] + ay[i+4] + ay[i+5] + ay[i+6] + ay[i+7] + ay[i+8] + ay[i+9]
                #          + ay[i+10] + ay[i+11] + ay[i+12] + ay[i+13] + ay[i+14] + ay[i+15]) / 16
                total = (ay[i] + ay[i + 1] + ay[i + 2] + ay[i + 3] + ay[i + 4]) / 5
                if total <= thresold:
                    sum = 0
                    it = 1
                    mean_value_50 = 0
                    for index in range(0, 100):
                        id = i + index
                        if id < length:
                            sum = sum + ay[id]
                            it = it + 1
                    if sum != 0:
                        mean_value_50 = sum / it
                    if mean_value_50 != 0 and mean_value_50 < thresold:
                        index = i
                        break
        return index

    def GetLeftAndRight(self, img):
        t1 = time.time()
        shape = img.shape
        height = shape[0]
        weight = shape[1]
        x_mid = int(weight / 2)
        pixel = img[0:height, x_mid-20:x_mid+20]
        pixel_mean = np.mean(pixel, axis=1)
        pix_mean_reverse = pixel_mean[::-1]
        reverse_y1 = self.GetIndex(pix_mean_reverse, 60, 0, height, True)
        y_max_mean = height - reverse_y1
        y_second_mean = height - self.GetIndex(pix_mean_reverse, 10, reverse_y1, height, False)
        nuclear_y = int((y_max_mean - y_second_mean) / 2)
        y_mid = int(y_second_mean + nuclear_y)

        #get left and right position of x
        pixel2 = img[y_mid:y_mid + nuclear_y, x_mid:weight]
        pixel2_mean = np.mean(pixel2, axis=0)
        x_right = x_mid + self.GetIndex(pixel2_mean, 10, 0, x_mid, False)

        pixel3 = img[y_mid:y_mid + nuclear_y, 0:x_mid]
        pixel3_mean = np.mean(pixel3, axis=0)
        pixel3_reverse = pixel3_mean[::-1]
        x_left = x_mid - self.GetIndex(pixel3_reverse, 10, 0, x_mid, False)

        t2 = time.time()
        print("get left and right position time:  " + str(t2-t1))
        return x_left, x_right

    def GetCompareResult(self, image):
        self.path = image
        img = cv2.imread(image, 0)
        self.height = img.shape[0]
        self.weight = img.shape[1]

        # img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 5)
        self.x_left, self.x_right = self.GetLeftAndRight(img)
        self.y_begin = int(self.height / 3.0 - 100)

        print('x left:  ' + str(self.x_left))
        print('x right:  ' + str(self.x_right))
        print('y position:  ' + str(self.y_begin))

        imgAry = np.asarray(Image.open(image))
        self.imageData = imgAry.copy()
        self.outputData = imgAry[int(self.y_begin):self.height, int(self.x_left):int(self.x_right)]

        img_data = np.zeros((1, 1024, 512, 3), np.uint8)
        size = (512, 1024)  # size w, hImage.fromarray(np.uint8(img))
        img2 = Image.fromarray(np.uint8(self.outputData)).resize(size)
        img_array = np.asarray(img2)
        img_data[0, :, :, :] = img_array

        return img_data
