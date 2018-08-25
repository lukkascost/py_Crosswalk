from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 4
ATT_NUMBER = 10
DECIMATION = 1
NUMBER_OF_ROUNDS = 50
CM_BIT_MIN = 2
CM_BIT_MAX = 8
R = 1
SHOW = True

oExp = Experiment.load(
    "OBJECTS/EXP_{:02d}/ACC_M{}_{}_CM{}-{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, DECIMATION,
                                                              NUMBER_OF_ROUNDS, CM_BIT_MIN, CM_BIT_MAX, ATT_NUMBER))
print oExp.show_in_table()

if SHOW:
    print "-" * 40
    oDataSetCm = oExp.experimentResults[R - 1]
    for j, i in enumerate(oDataSetCm.dataSet):
        print "Rodada ", j + 1
        print i
        print "-" * 40
