import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 4
ATT_NUMBER = 10
MIN_DECIMATION = 1
MAX_DECIMATION = 100
NUMBER_OF_ROUNDS = 50
CM_BITS = 8

M = 14

PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_04/'
NORMALIZATION_MATRIX = np.array([[9.98510800e-01, 2.25201230e-02],
                                 [5.46968100e+03, 1.38611180e+01],
                                 [9.99296500e-01, 5.01285000e-01],
                                 [2.15652630e+00, 3.52501050e-03],
                                 [9.99309000e-01, 5.40860700e-01],
                                 [2.07592710e+02, 8.86744200e-02],
                                 [9.99255060e-01, 1.01222746e-01],
                                 [2.88141800e+01, 7.24622100e-02],
                                 [9.68741850e-02, -8.40647000e-02],
                                 [4.80887420e+04, 5.70339160e+00]])
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
basemask = basemask - 1

svm = cv2.SVM()
oDataSet = DataSet()
base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH198.txt".format(M), usecols=basemask, delimiter=",")
classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH198.txt".format(M), dtype=object, usecols=24,
                     delimiter=",")
for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.attributes = (oDataSet.attributes - NORMALIZATION_MATRIX[:, 1].T) / (
        NORMALIZATION_MATRIX[:, 0] - NORMALIZATION_MATRIX[:, 1])
oData = Data(4, 13, samples=50)
svm.load("MODEL_M1_CM8_TH198_ATT10_ROUND_24.txt")
results = svm.predict_all(np.float32(oDataSet.attributes))
oData.set_results_from_classifier(results, oDataSet.labels)
oData.insert_model(svm)
print oData.confusion_matrix
print oData
