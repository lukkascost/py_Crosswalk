import numpy as np

from MachineLearn.Classes.experiment import Experiment

EXPERIMENT_NUMBER = 4
ATT_NUMBER = [3, 10, 24]
MIN_DECIMATION = 1
MAX_DECIMATION = 100
NUMBER_OF_ROUNDS = 50
CM_BITS = 8

oExp1 = Experiment.load(
    "../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, MIN_DECIMATION, MAX_DECIMATION,
                                                                 NUMBER_OF_ROUNDS, CM_BITS, ATT_NUMBER[0]))
oExp2 = Experiment.load(
    "../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, MIN_DECIMATION, MAX_DECIMATION,
                                                                 NUMBER_OF_ROUNDS, CM_BITS, ATT_NUMBER[0]))
oExp3 = Experiment.load(
    "../OBJECTS/EXP_{:02d}/ACC_M{}-{}_{}_CM{}b_ATT{}.txt".format(EXPERIMENT_NUMBER, MIN_DECIMATION, MAX_DECIMATION,
                                                                 NUMBER_OF_ROUNDS, CM_BITS, ATT_NUMBER[0]))
