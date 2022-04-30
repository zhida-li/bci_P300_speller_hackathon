"""
    @authors Zhida Li, Ana Laura Gonzalez Rios
    @email zhidal@sfu.ca
    @date Feb. 19, 2022
    @version: 1.1.0
    @description:
                This module contains implementation VFBLS and VCFBLS with datasets.

    @copyright Copyright (c) Feb. 19, 2022
        All Rights Reserved

    This Python code (versions 3.6 and newer)
"""

# ==============================================
# VFBLS and VCFBLS
# ==============================================
# Last modified: Apr. 30, 2022

# Import the built-in libraries
import os
import sys
import time
import random
import scipy.io
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import random
from scipy import signal
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# Import external libraries
import numpy as np
from scipy.stats import zscore

# Import customized libraries
# sys.path.append('../processing')
from bls.processing.replaceNan import replaceNan
from bls.model.vfbls_train_fscore import vfbls_train_fscore
from bls.model.vcfbls_train_fscore import vcfbls_train_fscore
from bls.processing.feature_select_cnl import feature_select_cnl

from sklearn.model_selection import train_test_split

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# blockPrint()

# ============================================== may edit --begin
# Load the datasets
mats = []
for i in range(1, 6):
    mats.append(scipy.io.loadmat(f'S{i}.mat'))

SAMPLING_FREQUENCY=250
PRE_SAMPLES = int(-0.1*SAMPLING_FREQUENCY) # -0.2
POST_SAMPLES = int(0.6*SAMPLING_FREQUENCY) # 0.4

p_samples = {}
n_samples = {}

for s in range(len(mats)):
    p_samples[s] = []
    n_samples[s] = []
    for i in range(len(mats[s]['trig'])):
        if mats[s]['trig'][i] == 0:
            pass #ignore this
        elif mats[s]['trig'][i] == 1:
            p_samples[s].append((i-PRE_SAMPLES, i+POST_SAMPLES))
        elif mats[s]['trig'][i] == -1:
            n_samples[s].append((i-PRE_SAMPLES, i+POST_SAMPLES))

# let's consider one subject for now, and then we can concat all together after if we want
# consider subject 0 for now
subject_index = 0
x = mats[subject_index]['y']

y = np.zeros(len(mats[subject_index]['trig']))
for start_i, end_i in p_samples[subject_index]:
    y[start_i:end_i] = 1

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)
# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
print(np.sum(train_y == 1), np.sum(test_y == 1))
# train_dataset = np.loadtxt("./slammer_64_train.csv", delimiter=",")
# test_dataset = np.loadtxt("./slammer_64_test.csv", delimiter=",")
# exit(0)
# Normalize training data
# train_x = train_dataset[:, 0:-1]
train_x = zscore(train_x, axis=0, ddof=1)  # For each feature, mean = 0 and std = 1
replaceNan(train_x)  # Replace "nan" with 0
# train_y = train_dataset[:, -1]

# Change training labels
inds1 = np.where(train_y == 0)
train_y[inds1] = 2

# Normalize test data
# test_x = test_dataset[:, 0:-1]
test_x = zscore(test_x, axis=0, ddof=1)  # For each feature, mean = 0 and std = 1
replaceNan(test_x)  # Replace "nan" with 0
# test_y = test_dataset[:, -1]

# Change test labels
inds1 = np.where(test_y == 0)
test_y[inds1] = 2

## VFBLS parameters
seed = 1  # set the seed for generating random numbers
num_class = 2  # number of the classes
epochs = 1  # number of epochs

C = 2 ** -15  # parameter for sparse regularization
s = 0.6  # the shrinkage parameter for enhancement nodes

#######################
# N1* - the number of mapped feature nodes
# N2* - the groups of mapped features
# N3* - the number of enhancement nodes

N1_bls_fsm = 100  # feature nodes of a group in the 1st set
N2_bls_fsm = 10  # feature groups in the 1st set

N1_bls_fsm1 = 20  # feature nodes of a group in the 2nd set
N2_bls_fsm1 = 10  # feature groups in the 3rd set

N1_bls_fsm2 = 20  # feature nodes of a group in the 3rd set
N2_bls_fsm2 = 10  # feature groups in the 3rd set

N3_bls_fsm = 100  # enhancement nodes

add_nFeature1 = 6  # no. of top relevant features in the 2nd set
add_nFeature2 = 4  # no. of top relevant features in the 3rd set
# ============================================== may edit --end

train_err = np.zeros((1, epochs))
test_err = np.zeros((1, epochs))
train_time = np.zeros((1, epochs))
test_time = np.zeros((1, epochs))

print("================== VFBLS ===========================\n\n")
np.random.seed(seed)  # set the seed for generating random numbers
for j in range(0, epochs):
    TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score \
        = vfbls_train_fscore(train_x, train_y, test_x, test_y, s, C,
                             N1_bls_fsm, N2_bls_fsm, N3_bls_fsm,
                             N1_bls_fsm1, N2_bls_fsm1,
                             N1_bls_fsm2, N2_bls_fsm2,
                             add_nFeature1, add_nFeature2)

    train_err[0, j] = TrainingAccuracy * 100
    test_err[0, j] = TestingAccuracy * 100
    train_time[0, j] = Training_time
    test_time[0, j] = Testing_time

blsfsm_test_acc = TestingAccuracy
blsfsm_test_f_score = f_score
blsfsm_train_time = Training_time
blsfsm_test_time = Testing_time

result = ['VFBLS', str(blsfsm_test_acc*100), str(blsfsm_test_f_score*100), str(blsfsm_train_time)]
result = np.asarray(result)
print('VFBLS results -Accuracy (%), F-Score (%), Training time (s)-:', result)
# np.savetxt('finalResult_VFBLS.csv', result, delimiter=',', fmt='%s')

##
print("================== VCFBLS ===========================\n\n")
np.random.seed(seed)  # set the seed for generating random numbers
for j in range(0, epochs):
    TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score \
        = vcfbls_train_fscore(train_x, train_y, test_x, test_y, s, C,
                              N1_bls_fsm, N2_bls_fsm, N3_bls_fsm,
                              N1_bls_fsm1, N2_bls_fsm1,
                              N1_bls_fsm2, N2_bls_fsm2,
                              add_nFeature1, add_nFeature2)

    train_err[0, j] = TrainingAccuracy * 100
    test_err[0, j] = TestingAccuracy * 100
    train_time[0, j] = Training_time
    test_time[0, j] = Testing_time

blsfsm_test_acc = TestingAccuracy
blsfsm_test_f_score = f_score
blsfsm_train_time = Training_time
blsfsm_test_time = Testing_time

result = ['VCFBLS', str(blsfsm_test_acc*100), str(blsfsm_test_f_score*100), str(blsfsm_train_time)]
result = np.asarray(result)
print('VCFBLS results -Accuracy (%), F-Score (%), Training time (s)-:', result)
# np.savetxt('finalResult_VCFBLS.csv', result, delimiter=',', fmt='%s')
