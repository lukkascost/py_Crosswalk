import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 50
PATH_TO_SAVE_FEATURES = '../../GLCM_FILES/EXP_04/'
NUMBER_OF_ROUNDS = 50
MIN_DECIMATION = 1
MAX_DECIMATION = 100

oExp = Experiment()
basemask = np.array(range(1,25))
svmVectors = []
basemask = basemask - 1

for M in range(MIN_DECIMATION, MAX_DECIMATION + 1):
    oDataSet = DataSet()
    base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH198.txt".format(M), usecols=basemask, delimiter=",")
    classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH198.txt".format(M), dtype=object, usecols=24,
                         delimiter=",")
    for x, y in enumerate(base):
        oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    oDataSet.normalize_data_set()
    for j in range(NUMBER_OF_ROUNDS):
        print j
        oData = Data(4, 13, samples=50)
        oData.random_training_test_per_class()
        svm = cv2.SVM()
        oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                            k_fold=2)
        svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                       np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
        svmVectors.append(svm.get_support_vector_count())
        results = svm.predict_all(np.float32(oDataSet.attributes))
        oData.set_results_from_classifier(results, oDataSet.labels)
        oData.insert_model(svm)
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet, description="  50 execucoes M={} CM=8b base CROSSWALK arquivos em EXP_04".format(M))
    print(oDataSet)
oExp.save("../../OBJECTS/EXP_05_ACC_M{}-{}_{}_CM8b_ATT24.txt".format(MIN_DECIMATION, MAX_DECIMATION, NUMBER_OF_ROUNDS))

######################

oExp = oExp.load("../../OBJECTS/EXP_05_ACC_M{}-{}_{}_CM8b_ATT24.txt".format(MIN_DECIMATION, MAX_DECIMATION, NUMBER_OF_ROUNDS))
print oExp.show_in_table()
