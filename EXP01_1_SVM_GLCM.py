import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 50
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_01/'
NUMBER_OF_ROUNDS = 50
MIN_DECIMATION = 1
MAX_DECIMATION = 10

oExp = Experiment()
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
svmVectors = []
basemask = basemask - 1


# for M in range(MIN_DECIMATION, MAX_DECIMATION + 1):
#     oDataSet = DataSet()
#     base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b.txt".format(M), usecols=basemask, delimiter=",")
#     classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b.txt".format(M), dtype=object, usecols=24, delimiter=",")
#     for x, y in enumerate(base):
#         oDataSet.addSampleOfAtt(np.array(list(np.float32(y)) + [classes[x]]))
#     oDataSet.atributes = oDataSet.atributes.astype(float)
#     oDataSet.normalizeDataSet()
#     for j in range(NUMBER_OF_ROUNDS):
#         print j
#         oData = Data(4, 13, samples=50)
#         oData.randomTrainingTestPerClass()
#         svm = cv2.SVM()
#         oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
#                             k_fold=2)
#         svm.train_auto(np.float32(oDataSet.atributes[oData.Training_indexes]),
#                        np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
#         svmVectors.append(svm.get_support_vector_count())
#         results = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))
#         oData.setResultsFromClassfier(results, oDataSet.labels[oData.Testing_indexes])
#         oDataSet.append(oData)
#     oExp.addDataSet(oDataSet, description="  50 execucoes M={} CM=8b base CROSSWALK arquivos em EXP_01".format(M))
#     print(oDataSet)
# oExp.save("OBJECTS/EXP_01_ACC_M1-10_50_CM8b.txt")

######################

oExp = oExp.load("OBJECTS/EXP_01_ACC_M1-10_50_CM8b.txt")
print oExp.show_in_table()