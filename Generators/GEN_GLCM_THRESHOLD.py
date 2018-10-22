import cv2
import numpy as np

from MachineLearn.Classes.Extractors.GLCM import GLCM

MIN_BITS = 8
MAX_BITS = 8

MIN_DECIMATION = 71
MAX_DECIMATION = 78

MIN_THRESHOLD_VALUE = 199
MAX_THRESHOLD_VALUE = 199
TH_STEP = 1

PATH_TO_IMAGES_FOLDER = '../../database-Crosswalk/Original/'
PATH_TO_SAVE_FEATURES = '../GLCM_FILES/EXP_06/'
for THRESHOLD in range(MIN_THRESHOLD_VALUE, MAX_THRESHOLD_VALUE + 1, TH_STEP):
    for nbits in range(MIN_BITS, MAX_BITS + 1):
        for k in range(MIN_DECIMATION, MAX_DECIMATION + 1):
            listGLCM = []
            for quantity in [[1, 150], [2, 150], [3, 150], [4, 150]]:
                for image in range(1, quantity[1] + 1):
                    img = cv2.imread(PATH_TO_IMAGES_FOLDER + "c{:d}_p1_{:d}.jpg".format(quantity[0], image), 0)
                    """ DECIMATION """
                    klist = [x for x in range(0, img.shape[0], k)]
                    klist2 = [x for x in range(0, img.shape[1], k)]
                    img = img[klist]
                    img = img[:, klist2]

                    """ APPLYING THRESHOLD ALGORITHM """
                    img[img <= THRESHOLD] = 0

                    """ CHANGING IMAGE TO VALUES BETWEEN 0 AND  2**NBITS"""
                    img = img / 2 ** (8 - nbits)

                    """ GENERATING FEATURES FOR GLCM """
                    oGlcm = GLCM(img, nbits)
                    oGlcm.generateCoOccurenceHorizontal()
                    oGlcm.normalizeCoOccurence()
                    oGlcm.calculateAttributes()

                    """ ADDING FEATURES IN ARRAY FOR SAVE IN FILE """
                    listGLCM.append(oGlcm.exportToClassfier("Class " + str(quantity[0])))
                    print nbits, k, quantity[0], image, THRESHOLD
            listGLCM = np.array(listGLCM)

            """ SAVE FILE WITH FEATURES, DECIMATION WITH STEP = k AND CORRELATION MATRIX WITH nbits BITS. """
            np.savetxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM{}b_TH{}.txt".format(k, nbits, THRESHOLD), listGLCM,
                       fmt="%s",
                       delimiter=',')
