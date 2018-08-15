from copy import copy

import cv2
import numpy as np

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.experiment import Experiment

oExp = Experiment.load("OBJECTS/EXPERIMENTO_04_MELHOR_TREINAMENTO_10000_CROSSWALK_198_LIMIAR.txt")
oDataSet = oExp.experimentResults[0]
for i, oData in enumerate(oDataSet.dataSet[:10]):
    svm = cv2.SVM()
    svm.load("SVM_MODELS/EXP_04/SVM_RBF_{}.gzip".format(i))
    results1 = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))

    oDatan = Data(4, 4)
    oDatan.Training_indexes = oData.Training_indexes
    oDatan.Testing_indexes = oData.Testing_indexes
    oDatan.insert_model(svm)
    oDatan.set_results_from_classifier(results1, oDataSet.labels[oData.Testing_indexes])
    oDataSet.dataSet[i] = copy(oDatan)
oExp.save("OBJECTS/EXPERIMENTO_04_MELHOR_TREINAMENTO_10000_CROSSWALK_198_LIMIAR.txt")
