import cv2
import numpy as np

from MachineLearn.Classes.Extractors.GLCM import GLCM

MIN_BITS = 8
MAX_BITS = 8

MIN_DECIMATION = 48
MAX_DECIMATION = 100

PATH_TO_IMAGES_FOLDER = '../database-Crosswalk/Original/'
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_02/'

listGLCM = []
for quantity in [[1, 50], [2, 50], [3, 50], [4, 150]]:
    for image in range(1, quantity[1] + 1):
        img = cv2.imread(PATH_TO_IMAGES_FOLDER + "c{:d}_p1_{:d}.jpg".format(quantity[0], image), 0)

        """ APPLYING OTSU'S ALGORITHM """
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        listGLCM.append(ret)

listGLCM = np.array(listGLCM)

print listGLCM
print np.mean(listGLCM), np.std(listGLCM)
print np.median(listGLCM)
print np.histogram(listGLCM)[0]
print np.histogram(listGLCM)[1]
