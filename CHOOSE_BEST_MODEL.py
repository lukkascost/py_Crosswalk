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


# ###############################################################################################################################
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
#     for j in range(10000):
#         oData = Data(4, 13, samples=50)
#         oData.randomTrainingTestPerClass()
#         svm = cv2.SVM()
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
#
# ################################################################################################################################
oExp = oExp.load("OBJECTS/EXPERIMENTO_04_MELHOR_TREINAMENTO_10000_CROSSWALK_198_LIMIAR.txt")
print oExp.experimentResults[0].dataSet[8872]
oData = oExp.experimentResults[0].dataSet[8872]
oDataSet = oExp.experimentResults[0]

print "Acuracias por classe: \n", oData.getMetrics().T
print "Matriz Confusao: \n", oData.confusion_matrix

svm = cv2.SVM()
svm.train(np.float32(oDataSet.atributes[oData.Training_indexes]), np.float32(oDataSet.labels[oData.Training_indexes]),
          params=oData.params)
print "Numero de vetores suporte: ", svm.get_support_vector_count()
svm.save("SVM_LUCAS10_1_THRES_198.txt")