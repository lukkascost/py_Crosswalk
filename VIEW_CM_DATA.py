from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 4
ATT_NUMBER = 3
DECIMATION = 14
DECIMATION_MIN = 1
DECIMATION_MAX = 100
NUMBER_OF_ROUNDS = 50
CM_BIT_MIN = 2
CM_BIT_MAX = 8
CM_BIT = 8

R = 5
SHOW = True

oExp = Experiment.load(
    "OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, DECIMATION_MIN,DECIMATION_MAX,
                                                              NUMBER_OF_ROUNDS, CM_BIT, ATT_NUMBER))
print oExp.show_in_table()

if SHOW:
    print "-" * 40
    oDataSetCm = oExp.experimentResults[R - 1]
    for j, i in enumerate(oDataSetCm.dataSet):
        print "Rodada ", j + 1
        print i
        print "-" * 40
oData = oDataSetCm.dataSet[23]
print oData
print "Matrix Confusao:"
print oData.confusion_matrix
print
print "Valores para Normalizacao(max, min):"
print oDataSetCm.normalize_between
print
print "Indices das amostras de teste(indexado de 0):"
print oData.Testing_indexes
# oData.save_model("MODEL_M14_CM6_TH198_ATT3_ROUND_24.txt")