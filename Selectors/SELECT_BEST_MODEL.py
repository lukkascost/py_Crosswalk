from MachineLearn.Classes.experiment import Experiment
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.data import Data
import numpy as np
import cv2

M = 1
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
TH = 198
ROUNDS = 10000

oExp = Experiment.load("../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
oDataSet = oExp.experimentResults[0]

bestIndexies = []
bestP1 = 0
bestP2 = 0
nvector = 1000000
i = 0
results = np.zeros((4, 4, 2))
for i, j in enumerate(oDataSet.dataSet):
    j.save_model("tmp.txt")
    svm = cv2.SVM()
    svm.load("tmp.txt")
    results1 = svm.predict_all(np.float32(oDataSet.attributes[j.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.attributes[j.Training_indexes]))
    acc = j.get_metrics()[0][-1]
    j.confusion_matrix = np.zeros((4, 4))
    j.set_results_from_classifier(results2, oDataSet.labels[j.Training_indexes])
    acc2 = j.get_metrics()[0, -1]
    if acc + acc2 >= bestP1:
        bestP1 = acc + acc2
        bestP2 = abs(acc - acc2)
        nvector = svm.get_support_vector_count()
        bestIndexies.append(i)
        print acc, acc2, nvector, bestP1, bestP2, i

bestIndexies = bestIndexies[-4:]
oExp = oExp.load("../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
oDataSet = oExp.experimentResults[0]

print "\nSIGMOID RESULTS"
for k in bestIndexies:
    oData = oDataSet.dataSet[k]
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_SIGMOID, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)

    results1 = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.attributes[oData.Training_indexes]))

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results1, oDataSet.labels[oData.Testing_indexes])
    acc = oData.get_metrics()[0][-1]

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results2, oDataSet.labels[oData.Training_indexes])
    acc2 = oData.get_metrics()[0][-1]
    print "{:04d}\t{:03.04f}\t{:03.04f}".format(k, acc * 100, acc2 * 100)

oExp = oExp.load("../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
oDataSet = oExp.experimentResults[0]
print "\nPOLY RESULTS"
for k in bestIndexies:
    oData = oDataSet.dataSet[k]
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_POLY, svm_type=cv2.SVM_C_SVC, degree=1, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)

    results1 = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.attributes[oData.Training_indexes]))

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results1, oDataSet.labels[oData.Testing_indexes])
    acc = oData.get_metrics()[0][-1]

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results2, oDataSet.labels[oData.Training_indexes])
    acc2 = oData.get_metrics()[0][-1]
    print "{:04d}\t{:03.04f}\t{:03.04f}".format(k, acc * 100, acc2 * 100)

oExp = oExp.load("../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
oDataSet = oExp.experimentResults[0]

print "\nLINEAR RESULTS"
for k in bestIndexies:
    oData = oDataSet.dataSet[k]
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, degree=1, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)

    results1 = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.attributes[oData.Training_indexes]))

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results1, oDataSet.labels[oData.Testing_indexes])
    acc = oData.get_metrics()[0][-1]

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results2, oDataSet.labels[oData.Training_indexes])
    acc2 = oData.get_metrics()[0][-1]
    print "{:04d}\t{:03.04f}\t{:03.04f}".format(k, acc * 100, acc2 * 100)

oExp = oExp.load("../OBJECTS/EXP_04/BEST_MODEL_{}_CROSSWALK_M{}_TH{}.txt".format(ROUNDS, M, TH))
oDataSet = oExp.experimentResults[0]

print "\nRBF RESULTS"
for k in bestIndexies:
    oData = oDataSet.dataSet[k]
    oData.save_model("tmp.txt")
    svm = cv2.SVM()
    svm.load("tmp.txt")

    results1 = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.attributes[oData.Training_indexes]))

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results1, oDataSet.labels[oData.Testing_indexes])
    acc = oData.get_metrics()[0][-1]

    oData.confusion_matrix = np.zeros((4, 4))
    oData.set_results_from_classifier(results2, oDataSet.labels[oData.Training_indexes])
    acc2 = oData.get_metrics()[0][-1]
    print "{:04d}\t{:03.04f}\t{:03.04f}".format(k, acc * 100, acc2 * 100)
