import os

from MachineLearn.Classes.experiment import Experiment
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.data import Data
import numpy as np
import cv2

M = 17
basemask = np.array([1, 2, 5, 9, 15, 16, 17, 21, 22, 23])
TH = 199
supportVectors = np.zeros((7, 50))

oExp = Experiment.load("../OBJECTS/EXP_06/ACC_M17-17_50_CM2-7b_TH199-199_ATT10.gzip")
for k in range(6):
    oDataSet = oExp.experimentResults[k]
    for i, j in enumerate(oDataSet.dataSet):
        j.save_model("tmp.txt")
        svm = cv2.SVM()
        svm.load("tmp.txt")
        supportVectors[k, i] = svm.get_support_vector_count()

oExp = Experiment.load("../OBJECTS/EXP_06/ACC_M1-100_50_CM8-8b_TH199-199_ATT10.gzip")
oDataSet = oExp.experimentResults[16]
for i, j in enumerate(oDataSet.dataSet):
    j.save_model("tmp.txt")
    svm = cv2.SVM()
    svm.load("tmp.txt")
    supportVectors[6, i] = svm.get_support_vector_count()
os.remove("tmp.txt")

medias = np.mean(supportVectors, axis=1)
desvios = np.std(supportVectors, axis=1)
minimo = np.min(supportVectors, axis=1)
maximo = np.max(supportVectors, axis=1)

print "\t\t\t{:02d}b\t\t{:02d}b\t\t{:02d}b\t\t{:02d}b\t\t{:02d}b\t\t{:02d}b\t\t{:02d}b\t\t".format(2, 3, 4, 5, 6, 7, 8)
str = "Medias: \t"
for i in medias:
    str+="{:03.2f}\t".format(i)
print str

str = "Desvio: \t"
for i in desvios:
    str+="{:03.2f}\t".format(i)
print str

str = "Minimos: \t"
for i in minimo:
    str+="{:03.2f}\t".format(i)
print str

str = "Maximos: \t"
for i in maximo:
    str+="{:03.2f}\t".format(i)
print str

