import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 150
PATH_TO_SAVE_FEATURES = '../../GLCM_FILES/EXP_06/'
NUMBER_OF_ROUNDS = 50
MIN_THRESHOLD = 181
MAX_THRESHOLD = 194
TH_STEP = 1
EXPERIMENT = 6

oExp = Experiment()
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
svmVectors = []
basemask = basemask - 1

for TH in range(MIN_THRESHOLD, MAX_THRESHOLD + 1, TH_STEP):
    oDataSet = DataSet()
    base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM8b_TH{}.txt".format(TH), usecols=basemask, delimiter=",")
    classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM8b_TH{}.txt".format(TH), dtype=object, usecols=24,
                         delimiter=",")
    for x, y in enumerate(base):
        oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    oDataSet.normalize_data_set()
    for j in range(NUMBER_OF_ROUNDS):
        print j, TH
        oData = Data(4, 50, samples=150)
        oData.random_training_test_per_class()
        svm = cv2.SVM()
        oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                            k_fold=10)
        svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                       np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
        svmVectors.append(svm.get_support_vector_count())
        results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
        oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
        oData.insert_model(svm)
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet,
                      description="  50 execucoes M=1 CM=8b base CROSSWALK arquivos em EXP_04 Threshold {} ".format(TH))
    print(oDataSet)
oExp.save(
    "../../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}-{}b_TH{}-{}_ATT{}.gzip".format(EXPERIMENT, 1, 1, NUMBER_OF_ROUNDS,
                                                                                8, 8, MIN_THRESHOLD, MAX_THRESHOLD, 10))

#####################

oExp = Experiment.load(
    "../../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}-{}b_TH{}-{}_ATT{}.gzip".format(EXPERIMENT, 1, 1, NUMBER_OF_ROUNDS, 8,
                                                                                8, MIN_THRESHOLD, MAX_THRESHOLD, 10))
print oExp.show_in_table()
