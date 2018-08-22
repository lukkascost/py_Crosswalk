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

possibles = []

avg_minus = np.mean(results[:5, 1]) - np.std(results[:5, 1])
avg_plus = np.mean(results[:5, 1]) + np.std(results[:5, 1])

for i in range(MIN_DECIMATION, MAX_DECIMATION + 1):
    if avg_minus <= results[i - MIN_DECIMATION, 0]:
        possibles.append(i)
print results[possibles]
print possibles

print avg_minus, avg_plus
