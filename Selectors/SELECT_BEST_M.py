import numpy as np

from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 4
ATT_NUMBER = 10
MIN_DECIMATION = 1
MAX_DECIMATION = 100
NUMBER_OF_ROUNDS = 50
CM_BITS = 8

oExp = Experiment.load(
    "../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, MIN_DECIMATION, MAX_DECIMATION,
                                                                 NUMBER_OF_ROUNDS, CM_BITS, ATT_NUMBER))
results = np.zeros((MAX_DECIMATION - MIN_DECIMATION + 1, 3))
for i, oDataSet in enumerate(oExp.experimentResults):
    metrics = oDataSet.get_general_metrics()
    results[i, 0] = metrics[0][0, -1] - metrics[1][0, -1]
    results[i, 1] = metrics[0][0, -1]
    results[i, 2] = metrics[0][0, -1] + metrics[1][0, -1]

results = results * 100

possibles_a = []
possibles_b = []
possibles_c = []
avg_minus = np.min(results[:3, 0])

for i in range(MIN_DECIMATION, MAX_DECIMATION + 1):
    if avg_minus <= results[i - MIN_DECIMATION, 0]:
        possibles_a.append(i)
    if avg_minus <= results[i - MIN_DECIMATION, 1]:
        possibles_b.append(i)
    if avg_minus <= results[i - MIN_DECIMATION, 2]:
        possibles_c.append(i)


print "CASO 1: Media"
print "\tPossiveis: ", possibles_b
print "M{}".format(possibles_b[-1])
print "\tacc-std = {}".format(results[possibles_b[-1] - 1, 0])
print "\tacc     = {}".format(results[possibles_b[-1] - 1, 1])
print "\tacc+std = {}".format(results[possibles_b[-1] - 1, 2])
print
print "CASO 2: Media - std"
print "\tPossiveis: ", possibles_a
print "M{}".format(possibles_a[-1])
print "\tacc-std = {}".format(results[possibles_a[-1] - 1, 0])
print "\tacc     = {}".format(results[possibles_a[-1] - 1, 1])
print "\tacc+std = {}".format(results[possibles_a[-1] - 1, 2])
print
print "CASO 1: Media + std"
print "\tPossiveis: ", possibles_c
print "M{}".format(possibles_c[-1])
print "\tacc-std = {}".format(results[possibles_c[-1] - 1, 0])
print "\tacc     = {}".format(results[possibles_c[-1] - 1, 1])
print "\tacc+std = {}".format(results[possibles_c[-1] - 1, 2])
