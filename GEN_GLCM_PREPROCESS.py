import cv2
import numpy as np

from MachineLearn.Classes.Extractors.GLCM import GLCM

for nbits in range(2, 9):
    for k in range(1, 11):
        listGLCM = []
        for quantity in [[1, 50], [2, 50], [3, 50], [4, 150]]:
            for image in range(1, quantity[1]+1):
                img = cv2.imread("../database-Crosswalk/Preprocessed/c{:d}_p1_{:d}.JPG".format(quantity[0], image), 0)

                """ DECIMATION """
                klist = [x for x in range(0, img.shape[0], k)]
                klist2 = [x for x in range(0, img.shape[1], k)]
                img = img[klist]
                img = img[:, klist2]

                """ CHANGING IMAGE TO VALUES BETWEEN 0 AND  2**NBITS"""
                img = img / 2 ** (8 - nbits)

                """ GENERATING FEATURES FOR GLCM """
                oGlcm = GLCM(img, nbits)
                oGlcm.generateCoOccurenceHorizontal()
                oGlcm.normalizeCoOccurence()
                oGlcm.calculateAttributes()

                """ ADDING FEATURES IN ARRAY FOR SAVE IN FILE """
                listGLCM.append(oGlcm.exportToClassfier("Class " + str(quantity[0])))
                print nbits, k, quantity[0], image
        listGLCM = np.array(listGLCM)

        """ SAVE FILE WITH FEATURES, DECIMATION WITH STEP = k AND CORRELATION MATRIX WITH nbits BITS. """
        np.savetxt("GLCM_FILES/EXP_01/FEATURES_M{}_CM{}b.txt".format(k, nbits), listGLCM, fmt="%s", delimiter=',')
