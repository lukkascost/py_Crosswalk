import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 50
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_02/'
NUMBER_OF_ROUNDS = 50
MIN_BITS = 2
MAX_BITS = 8

oExp = Experiment()
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
svmVectors = []
basemask = basemask - 1


# for n_bits in range(MIN_BITS, MAX_BITS + 1):
#     oDataSet = DataSet()
#     base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM{}b.txt".format(n_bits), usecols=basemask, delimiter=",")
#     classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM{}b.txt".format(n_bits), dtype=object, usecols=24, delimiter=",")
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
#     oExp.addDataSet(oDataSet, description="  50 execucoes M=1 CM={}b base CROSSWALK arquivos em EXP_02".format(n_bits))
#     print(oDataSet)
# oExp.save("OBJECTS/EXP_02_ACC_M1_50_CM2-8b.txt")

######################

oExp = oExp.load("OBJECTS/EXP_02_ACC_M1_50_CM2-8b.txt")
print oExp