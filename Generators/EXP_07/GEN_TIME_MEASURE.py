import numpy as np
import time as tm
import cv2
from MachineLearn.Classes.Extractors.GLCM import GLCM

gerarTabelas = True
gerarResultados = True
PATH_TO_IMAGES_FOLDER = '../../../database-Crosswalk/Original/'
PATH_TO_SAVE_RESULTS = '../../TIMES/EXP_06/'
NATT = 10
M_CM = 8
CM_M = 17


def measureTime10(img, bits=6):
    """
    Input data
    """
    ########################################################################
    img = img / 2 ** (8 - bits)
    oGlcm = GLCM(img, bits, number_of_Attributes=NATT)
    """10 Atts"""
    basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])

    """24 atts"""
    # basemask = np.array(range(1, 25))
    #
    """3 Atts """
    # basemask = np.array([7, 12, 22])

    basemask = basemask - 1
    times = np.zeros(4)
    glcm_atributes = np.zeros(25, dtype=np.float64)
    gray = 2 ** bits
    ########################################################################

    start = tm.time()
    oGlcm.generateCoOccurenceHorizontal()
    end = tm.time()
    times[0] = end - start
    ########################################################################

    start = tm.time()
    oGlcm.normalizeCoOccurence()
    end = tm.time()
    times[1] = end - start
    ########################################################################

    start = tm.time()
    """10 atts"""
    for i in range(gray):
        for j in range(gray):
            ij = oGlcm.coOccurenceNormalized[i, j]
            glcm_atributes[1] += ij * ij
            glcm_atributes[2] += ((i - j) * (i - j) * (ij))
            glcm_atributes[5] += (ij) / (1 + ((i - j) * (i - j)))
            glcm_atributes[9] += ij * np.log10(ij + 1e-30)
            glcm_atributes[15] += (ij) / (1 + abs(i - j))
            glcm_atributes[16] += ij * (i + j)
            glcm_atributes[21] += ij * abs(i - j)
            glcm_atributes[22] += ij * (i - j)
            glcm_atributes[23] += ij * i * j
    glcm_atributes[17] = np.amax(oGlcm.coOccurenceNormalized)
    glcm_atributes[16] /= 2
    glcm_atributes[22] /= 2
    glcm_atributes[9] *= -1

    """24 Atts"""
    # oGlcm.calculateAttributes()

    """3 Atts"""
    # gray = oGlcm.coOccurenceMatrix.shape[0]
    # HXY1 = 0.0
    # HXY2 = 0.0
    # HX = 0.0
    # HY = 0.0
    #
    # px = np.zeros(gray)
    # py = np.zeros(gray)
    # px_plus_y = np.zeros(gray * 2 - 1)
    # px_minus_y = np.zeros(gray)
    # for i in range(gray):
    #     px[i] = sum(oGlcm.coOccurenceNormalized[i, :])
    #     py[i] = sum(oGlcm.coOccurenceNormalized[:, i])
    #     for j in range(gray):
    #         Pij = oGlcm.coOccurenceNormalized[i, j]
    #         px_plus_y[i + j] += oGlcm.coOccurenceNormalized[i, j]
    #         px_minus_y[abs(i - j)] += Pij
    #         glcm_atributes[9] += Pij * np.log10(Pij + 1e-30)
    #         glcm_atributes[22] += Pij * (i - j)
    # glcm_atributes[22] /= 2
    #
    # Q = np.zeros((gray, gray))
    # for i in range(gray * 2 - 1):
    #     glcm_atributes[8] += px_plus_y[i] * np.log10(px_plus_y[i] + 1e-30)
    # for i in range(gray):
    #     HX += px[i] * np.log10(px[i] + 1e-30)
    #     HY += py[i] * np.log10(py[i] + 1e-30)
    #     for j in range(gray):
    #         HXY1 += oGlcm.coOccurenceNormalized[i, j] * np.log10(px[i] * py[j] + 1e-30)
    #         HXY2 += px[i] * py[j] * np.log10(px[i] * py[j] + 1e-30)
    #         for k in range(gray):
    #             Q[i, j] = oGlcm.coOccurenceNormalized[i, k] * oGlcm.coOccurenceNormalized[j, k]
    #             if not (Q[i, j] == 0):
    #                 Q[i, j] = Q[i, j] / (px[i] * py[k])
    #
    # glcm_atributes[8] *= -1
    # glcm_atributes[9] *= -1
    # HXY1 *= -1
    # HXY2 *= -1
    #
    # glcm_atributes[12] = glcm_atributes[9] - HXY1
    # m1 = np.amax(HX)
    # m2 = np.amax(HY)
    # if m1 > m2:
    #     glcm_atributes[12] /= m1
    # else:
    #     glcm_atributes[12] /= m2
    #
    # for i in range(gray * 2 - 1):
    #     glcm_atributes[7] += pow(i - glcm_atributes[8], 2) * px_plus_y[i]
    # oGlcm.attributes = glcm_atributes[1:]


    end = tm.time()
    times[2] = end - start

    ########################################################################

    oGlcm.setAtributesValues(glcm_atributes[basemask])
    svm = cv2.SVM()
    svm.load("MODEL_M17_CM8_TH199_ATT{}_ROUND_0.txt".format(NATT))

    start = tm.time()
    svm.predict(np.float32(oGlcm.attributes))
    end = tm.time()
    times[3] = end - start
    ########################################################################
    return times * 1000


if gerarResultados:
    imgs_times = np.zeros((3, 5))
    # for k, m in enumerate([27, 17, 1]):
    #     total_times = []
    #     for i in range(100):
    #         img = cv2.imread(PATH_TO_IMAGES_FOLDER + "c{:d}_p1_{:d}.jpg".format(1, 1), 0)
    #
    #         """ DECIMATION """
    #         klist = [x for x in range(0, img.shape[0], m)]
    #         klist2 = [x for x in range(0, img.shape[1], m)]
    #         img = img[klist]
    #         img = img[:, klist2]
    #
    #         """ APPLYING THRESHOLD ALGORITHM """
    #         img[img <= 199] = 0
    #
    #         total_times.append(measureTime10(img, bits=M_CM))
    #         print total_times[-1], i
    #     total_times = np.array(total_times)
    #     imgs_times[k, :4] = np.min(total_times, axis=0)
    #     imgs_times[k, -1] = np.sum(np.min(total_times, axis=0))
    #     print imgs_times
    #     np.savetxt(PATH_TO_SAVE_RESULTS + "T2_M27,17,1_CM8b_ATT{}.txt".format(NATT), imgs_times, delimiter=',')
    #     np.savetxt(PATH_TO_SAVE_RESULTS + "T2_M27,17,1_CM8b_Line_{:03d}_ATT{}.txt".format(k, NATT), total_times,
    #                delimiter=',')

    imgs_times = np.zeros((7, 5))
    for l, b in enumerate(range(2, 9)):
        total_times = []
        for i in range(100):
            img = cv2.imread(PATH_TO_IMAGES_FOLDER + "c{:d}_p1_{:d}.jpg".format(1, 1), 0)
            """ DECIMATION """
            klist = [x for x in range(0, img.shape[0], CM_M)]
            klist2 = [x for x in range(0, img.shape[1], CM_M)]
            img = img[klist]
            img = img[:, klist2]

            """ APPLYING THRESHOLD ALGORITHM """
            img[img <= 198] = 0

            total_times.append(measureTime10(img, bits=b))
            print total_times[-1], i
        total_times = np.array(total_times)
        imgs_times[l, :4] = np.min(total_times, axis=0)
        imgs_times[l, -1] = np.sum(np.min(total_times, axis=0))
        print imgs_times
        np.savetxt(PATH_TO_SAVE_RESULTS + "T1_M1_CM2-8b_ATT{}.txt".format(NATT), imgs_times, delimiter=',')
        np.savetxt(PATH_TO_SAVE_RESULTS + "T1_M1_CM2-8b_Line_{:03d}_ATT{}.txt".format(l, NATT), total_times,
                   delimiter=',')

if (gerarTabelas):
    ### T2_M100,50,10,1_CM5b
    print "{:#^80}".format(" T2_M27,17,1_CM8b ")
    strResult = ""
    for k in range(4):
        for i, j in enumerate([27, 17, 1]):
            totals = np.loadtxt(PATH_TO_SAVE_RESULTS + "T2_M27,17,1_CM8b_Line_{:03d}_ATT{}.txt".format(i, NATT),
                                delimiter=",")
            strResult += str(round(np.average(totals, axis=0)[k], 4)) + "+-"
            strResult += str(round(np.std(totals, axis=0)[k], 4)) + "\t"
        strResult += "\n"

    for i, j in enumerate([27, 17, 1]):
        totals = np.loadtxt(PATH_TO_SAVE_RESULTS + "T2_M27,17,1_CM8b_Line_{:03d}_ATT{}.txt".format(i, NATT),
                            delimiter=",")
        strResult += str(np.round(np.sum(np.average(totals, axis=0)), 4)) + "+-"
        strResult += str(np.round(np.sum(np.std(totals, axis=0)), 4)) + "\t"
    print strResult

    print "{:#^80}".format("")

    ### T1_M1_CM1-8b
    print "{:#^80}".format(" T1_M1_CM2-8b ")
    strResult = ""
    for k in range(4):
        for i, j in enumerate([2, 3, 4, 5, 6, 7, 8]):
            totals = np.loadtxt(PATH_TO_SAVE_RESULTS + "T1_M1_CM2-8b_Line_{:03d}_ATT{}.txt".format(i, NATT),
                                delimiter=",")
            strResult += str(round(np.average(totals, axis=0)[k], 4)) + "+-"
            strResult += str(round(np.std(totals, axis=0)[k], 4)) + "\t"
        strResult += "\n"

    for i, j in enumerate([2, 3, 4, 5, 6, 7, 8]):
        totals = np.loadtxt(PATH_TO_SAVE_RESULTS + "T1_M1_CM2-8b_Line_{:03d}_ATT{}.txt".format(i, NATT), delimiter=",")
        strResult += str(np.round(np.sum(np.average(totals, axis=0)), 4)) + "+-"
        strResult += str(np.round(np.sum(np.std(totals, axis=0)), 4)) + "\t"
    print strResult

    print "{:#^80}".format("")
