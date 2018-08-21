from MachineLearn.Classes.experiment import Experiment
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.data import Data
import numpy as np
import cv2

oExp = Experiment()
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
acuracia = 0
nVetores = 10000
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_04/'
TH = 198

###############################################################################################################################
# basemask = basemask - 1
# best = 0
#
# for i in range(1):
#     oDataSet = DataSet()
#     base = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM8b_TH{}.txt".format(TH), usecols=basemask, delimiter=",")
#     classes = np.loadtxt(PATH_TO_SAVE_FEATURES + "FEATURES_M1_CM8b_TH{}.txt".format(TH), dtype=object, usecols=24, delimiter=",")
#     for x, y in enumerate(base):
#         oDataSet.addSampleOfAtt(np.array(list(np.float32(y)) + [classes[x]]))
#     oDataSet.atributes = oDataSet.atributes.astype(float)
#     oDataSet.normalizeDataSet()
#     for j in range(10):
#         oData = Data(4, 13, samples=50)
#         oData.randomTrainingTestPerClass()
#         svm = oData.svm
#         oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
#                             k_fold=2)
#         svm.train_auto(np.float32(oDataSet.atributes[oData.Training_indexes]),
#                        np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
#         results = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))
#         oData.setResultsFromClassfier(results, oDataSet.labels[oData.Testing_indexes])
#         if oData.getMetrics()[0, -1] > acuracia:
#             nVetores = svm.get_support_vector_count()
#             acuracia = oData.getMetrics()[0, -1]
#             print nVetores, acuracia, j
#             best = j
#         elif oData.getMetrics()[0, -1] == acuracia:
#             if nVetores > svm.get_support_vector_count():
#                 nVetores = svm.get_support_vector_count()
#                 best = j
#
#         oDataSet.append(oData)
#     oExp.addDataSet(oDataSet,
#                     description="10000 execucoes para escolher melhores vetores de suporte, o melhor esta no index {:05d}".format(
#                         best))
# oExp.save("OBJECTS/EXPERIMENTO_04_MELHOR_TREINAMENTO_10000_CROSSWALK_198_LIMIAR.txt")

# ################################################################################################################################
oExp = oExp.load("OBJECTS/EXPERIMENTO_04_MELHOR_TREINAMENTO_10000_CROSSWALK_198_LIMIAR.txt")
# print oExp.experimentResults[0].dataSet[8872]

oDataSet = oExp.experimentResults[0]
#
# bestIndexies = []
# bestP1 = 0
# bestP2 = 0
# nvector = 1000000
# i = 0
# for oData in oDataSet.dataSet[i:]:
#     svm = None
#     svm = cv2.SVM()
#     svm.train_auto(np.float32(oDataSet.atributes[oData.Training_indexes]),
#                    np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
#     svm.save("SVM_MODELS/EXP_04/SVM_RBF_{}.gzip".format(i))
#     results1 = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))
#     results2 = svm.predict_all(np.float32(oDataSet.atributes[oData.Training_indexes]))
#
#     oData.confusion_matrix = np.zeros((4, 4))
#     oData.setResultsFromClassfier(results1, oDataSet.labels[oData.Testing_indexes])
#     acc = oData.getMetrics()[0, -1]
#
#     oData.confusion_matrix = np.zeros((4,4))
#     oData.setResultsFromClassfier(results2, oDataSet.labels[oData.Training_indexes])
#     acc2 = oData.getMetrics()[0, -1]
#     if acc + acc2 >= bestP1:
#         if True: #abs(acc - acc2) >= bestP2:
#             # print acc, acc2, nvector,  acc + acc2, abs(acc - acc2), i, "M2"
#             if True: #svm.get_support_vector_count() <= nvector:
#                 bestP1 = acc + acc2
#                 bestP2 = abs(acc - acc2)
#                 nvector = svm.get_support_vector_count()
#                 bestIndexies.append(i)
#                 print acc, acc2, nvector, bestP1, bestP2, i
#
#     i += 1
#     # print "Numero de vetores suporte: ", svm.get_support_vector_count()
firsts = [8872, 7516, 6041, 2377, 1324]
for i in firsts:
    oData = oExp.experimentResults[0].dataSet[i]
    svm = cv2.SVM()
    svm.load("SVM_MODELS/EXP_04/SVM_RBF_{}.gzip".format(i))
    results1 = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))
    results2 = svm.predict_all(np.float32(oDataSet.atributes[oData.Training_indexes]))
    oData.confusion_matrix = np.zeros((4, 4))
    oData.setResultsFromClassfier(results2, oDataSet.labels[oData.Training_indexes])
    acc = oData.getMetrics()[0, -1]

    print i
    print oData
    print