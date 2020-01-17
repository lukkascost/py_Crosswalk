import cv2
import numpy as np

from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 7
ATT_NUMBER = 10
DECIMATION = 17
DECIMATION_MIN = 1
DECIMATION_MAX = 100
NUMBER_OF_ROUNDS = 50
CM_BIT_MIN = 8
CM_BIT_MAX = 8
CM_BIT = 8
TH_MIN = 199
TH_MAX = 199

R = 17
ROUND = 0
SHOW = True

oExp = Experiment.load(
    "OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}-{}b_TH{}-{}_ATT{}.gzip".format(EXPERIMENT_NUMBER, DECIMATION_MIN,
                                                                         DECIMATION_MAX,
                                                                         NUMBER_OF_ROUNDS, CM_BIT_MIN, CM_BIT_MAX,
                                                                         TH_MIN, TH_MAX,
                                                                         ATT_NUMBER))
print (oExp.show_in_table())

if SHOW:
    print( "-" * 40)
    oDataSetCm = oExp.experimentResults[R - 1]
    for j, i in enumerate(oDataSetCm.dataSet):
        print("Rodada ", j + 1)
        print(i)
        print("-" * 40)
oData = oDataSetCm.dataSet[ROUND - 1]
print(oDataSetCm)
print("Matrix Confusao:")
print(oData.confusion_matrix)
print()
print("Valores para Normalizacao(max, min):")
print(oDataSetCm.normalize_between)
print()
print("Indices das amostras de teste(indexado de 0):")
print(oData.Testing_indexes)
print(oDataSetCm)
# oData.save_model("MODEL_M{}_CM{}_TH{}_ATT{}_ROUND_{}.txt".format(DECIMATION, CM_BIT, TH_MIN, ATT_NUMBER, ROUND))
# svm = cv2.SVM()
# indices = [129, 142, 140, 104, 133, 114, 110, 143, 122, 148,
#            101, 115, 127]
# svm.load("MODEL_M{}_CM{}_TH{}_ATT{}_ROUND_{}.txt".format(DECIMATION, CM_BIT, 198, ATT_NUMBER, ROUND))
#
# attrs = [0.4338595, 0.1854257, 0.6563919, 0.5330247, 0.6618491, 0.3360929, 0.6377692, 0.2019472, 0.6526155, 0.2889805]
# print oDataSetCm.attributes[114]
# print svm.predict(np.float32(attrs))
