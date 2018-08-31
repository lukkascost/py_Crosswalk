import cv2

import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

SAMPLES_PER_CLASS = 50
PATH_TO_SAVE_FEATURES = '../../GLCM_FILES/EXP_04/'
NUMBER_OF_ROUNDS = 50
MIN_BITS = 2
MAX_BITS = 8
DECIMATION = 14

oExp = Experiment()
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
svmVectors = []
basemask = basemask - 1

for n_bits in range(MIN_BITS, MAX_BITS + 1):
    oDataSet = DataSet()
    base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM{}b_TH198.txt".format(DECIMATION, n_bits),
                      usecols=basemask, delimiter=",")
    classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM{}b_TH198.txt".format(DECIMATION, n_bits),
                         dtype=object, usecols=24, delimiter=",")
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
        oData.insert_model(svm)
        svmVectors.append(svm.get_support_vector_count())
        results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
        oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
        oDataSet.append(oData)
    oExp.add_data_set(oDataSet,
                      description="  50 execucoes M={} CM={}b base CROSSWALK arquivos em EXP_04".format(DECIMATION,n_bits))
    print(oDataSet)
oExp.save(
    "../../OBJECTS/EXP_{:02d}/ACC_M{}_{}_CM{}-{}b_ATT10.txt".format(4, DECIMATION, NUMBER_OF_ROUNDS, MIN_BITS, MAX_BITS))

######################

oExp = oExp.load(
    "../../OBJECTS/EXP_{:02d}/ACC_M{}_{}_CM{}-{}b_ATT10.txt".format(4, DECIMATION, NUMBER_OF_ROUNDS, MIN_BITS, MAX_BITS))
print oExp.show_in_table()
