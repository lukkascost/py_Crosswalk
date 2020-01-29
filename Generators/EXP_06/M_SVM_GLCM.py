import cv2

import numpy as np
import cv2.ml as ml

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 150
PATH_TO_SAVE_FEATURES = '../../GLCM_FILES/EXP_06/'
NUMBER_OF_ROUNDS = 50
MIN_DECIMATION = 1
MAX_DECIMATION = 1
EXPERIMENT = 6
NBITS = 8
NATT = 24

oExp = Experiment()
# basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
# basemask = np.array([12, 20, 22])
basemask = np.array(range(1, 25))
svmVectors = []
basemask = basemask - 1

for M in range(MIN_DECIMATION, MAX_DECIMATION + 1):
    oDataSet = DataSet()
    base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH199.txt".format(M), usecols=basemask, delimiter=",")
    classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH199.txt".format(M), dtype=object, usecols=24,
                         delimiter=",")
    for x, y in enumerate(base):
        oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    oDataSet.normalize_data_set()
    for j in range(NUMBER_OF_ROUNDS):
        print(j)
        oData = Data(4, 50, samples=150)
        oData.random_training_test_per_class()
        svm = ml.SVM_create()
        svm.setKernel(ml.SVM_RBF)
        oData.params = dict(kernel_type=ml.SVM_RBF, svm_type=ml.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                            k_fold=10)
        svm.trainAuto(np.float32(oDataSet.attributes[oData.Training_indexes]), ml.ROW_SAMPLE,
                      np.int32(oDataSet.labels[oData.Training_indexes]), kFold=10)
        # svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
        #                np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
        svmVectors.append(svm.getSupportVectors().shape[0])
        results = []  # svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
        for i in (oDataSet.attributes[oData.Testing_indexes]):
            res, cls = svm.predict(np.float32([i]))
            results.append(cls[0])
        oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
        oData.insert_model(svm)
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet,
                      description="  50 execucoes M={} CM={}b base CROSSWALK arquivos em EXP_{:02d}".format(M, NBITS,
                                                                                                            EXPERIMENT))
    print(oDataSet)
    oExp.save("../../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}-{}b_TH{}-{}_ATT{}.gzip".format(EXPERIMENT, MIN_DECIMATION,
                                                                                          MAX_DECIMATION,
                                                                                          NUMBER_OF_ROUNDS,                                                                                          NBITS, NBITS, 199, 199, NATT))

######################

oExp = oExp.load("../../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}-{}b_TH{}-{}_ATT{}.gzip".format(EXPERIMENT, MIN_DECIMATION,
                                                                                      MAX_DECIMATION, NUMBER_OF_ROUNDS,
                                                                                      NBITS, NBITS, 199, 199, NATT))
print(oExp.show_in_table())
