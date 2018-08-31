from MachineLearn.Classes.experiment import Experiment
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.data import Data
import numpy as np
import cv2

oExp = Experiment()
M = 1
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
accuracy = 0
nVectors = 10000
PATH_TO_SAVE_FEATURES = '../../GLCM_FILES/EXP_04/'
TH = 198
ROUNDS = 10000
basemask = basemask - 1
best = 0

for i in range(1):
    oDataSet = DataSet()
    base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH{}.txt".format(M, TH), usecols=basemask,
                      delimiter=",")
    classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM8b_TH{}.txt".format(M, TH), dtype=object, usecols=24,
                         delimiter=",")
    for x, y in enumerate(base):
        oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
    oDataSet.attributes = oDataSet.attributes.astype(float)
    oDataSet.normalize_data_set()
    for j in range(ROUNDS):
        oData = Data(4, 13, samples=50)
        oData.random_training_test_per_class()
        svm = cv2.SVM()
        oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                            k_fold=2)
        svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                       np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
        oData.insert_model(svm)
        results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
        oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
        if oData.get_metrics()[0, -1] > accuracy:
            nVectors = svm.get_support_vector_count()
            accuracy = oData.get_metrics()[0, -1]
            print nVectors, accuracy, j
            best = j
        elif oData.get_metrics()[0, -1] == accuracy:
            if nVectors > svm.get_support_vector_count():
                nVectors = svm.get_support_vector_count()
                best = j

        oDataSet.append(oData)
    oExp.add_data_set(oDataSet,
                      description=
                      "{} execucoes para escolher melhores vetores de suporte, o melhor esta no index {:05d}".format(
                          ROUNDS,
                          best))
oExp.save("../../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
