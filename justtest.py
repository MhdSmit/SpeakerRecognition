from scipy.io import wavfile
import numpy as np
import python_speech_features
"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import pickle
from sklearn.preprocessing import LabelBinarizer
"""

np.random.seed(1)

# Sounds
sdir = "./ahlam data/"
NU = 10  # number of speaker
# MFCC parameters
NC = 13  # number of coeff
NF = 44  # number of filters
NP = 1024  # number of samples in a frame (128 ms)
INC = 512  # increment between frames
NCC = 3 * NC  # total number of coefficients
# Speakers cof
NT=50  #number of frames for testing from each user
DF = NT * NU  # all data frames
NTT = 1900
DFF = NTT * NU
SP = np.zeros([NU, NCC,NTT])  # Features( NU: num of users, NCC: Num of MFCC + Deltas, NTT: Number of frames extracted from  MFCC
SP2 = np.zeros([NU, NCC, NT])  # same as previous but for test

for i in range(NU):
    # print(i)
    fname2 = sdir + 'test/speaker-' + str(i + 1) + '.wav'
    fname = sdir + 'train/speaker-' + str(i + 1) + '.wav'

    # print(fname)
    (fs, data) = wavfile.read(fname)
    (fs2, data2) = wavfile.read(fname2)
    # print(np.shape(data))
    # data = data / np.max(np.abs(data))
    # data2 = data2 / np.max(np.abs(data2))
    mfcc1 = python_speech_features.base.mfcc(data, samplerate=fs, winlen=0.128, winstep=0.064, numcep=NC + 1,
                                                 nfilt=NF,
                                                 nfft=NP, preemph=0, appendEnergy=False, winfunc=np.hamming)

    mfcc2 = python_speech_features.base.mfcc(data2, samplerate=fs2, winlen=0.128, winstep=0.064, numcep=NC + 1,
                                                 nfilt=NF,
                                                 nfft=NP, preemph=0, appendEnergy=False, winfunc=np.hamming)
    mfcc = mfcc1[:, 1:]  # remove first column coeff 0

    mfcc0 = mfcc2[:, 1:]  # remove first column coeff 0
    delta = python_speech_features.base.delta(mfcc, 1)

    delta2 = python_speech_features.base.delta(mfcc0, 1)
    delta_delta = python_speech_features.base.delta(delta, 1)

    delta2_delta = python_speech_features.base.delta(delta2, 1)
    cc = np.concatenate((mfcc.T, delta.T, delta_delta.T), axis=0)

    cc2 = np.concatenate((mfcc0.T, delta2.T, delta2_delta.T), axis=0)
    #print("cc is/n")
    #print(cc)
    #print("cc2 is/n")
    #print(cc2)
    temp = cc[:, 0:NTT]
    #print("temp is")
    #print(temp)
    temp2 = cc2[:, 0:NT]
    SP[i, :, :] = temp
    SP2[i, :, :] = temp2
    #print("SP is")
    #print(SP)


def get_train_X(SP):
    ret = np.zeros((DFF, NCC))
    for u in range(len(SP)):
        temp = SP[u].T
        for i in range(NTT):
            ret[NTT * u + i] = temp[i]

    return ret


def get_test_X(SP):
    ret = np.zeros((DF, NCC))
    for u in range(len(SP)):
        temp = SP[u].T
        for i in range(NT):
            ret[NT * u + i] = temp[i]

    return ret


def get_train_Y():
    rett = []
    for i in range(NU):
        for j in range(NTT):
            ret = np.zeros((NU))
            ret[i] = 1;
            rett.append(ret)
    return np.asarray(rett)
ss=get_train_X(SP)
ss1=get_train_Y()
fff1 = open("ann1.txt", "a")
fff1.write("===")
fff1.write("cc is\n")
fff1.write(str(cc))
fff1.write("\n")
fff1.write("temp is\n")
fff1.write(str(temp))
fff1.write("\n")
fff1.write("sp is\n")
fff1.write(str(SP))
fff1.write("\n")
fff1.write("ss is\n")
fff1.write(str(ss))
fff1.write("\n")
fff1.write("ss1 is\n")
fff1.write(str(ss1))
fff1.write("\n")
fff1.close()

def get_test_Y():
    rett = []

    for i in range(NU):
        for j in range(NT):
            ret = np.zeros((NU))
            ret[i] = 1;
            rett.append(ret)

    return np.asarray(rett)


def get_data_Y():
    ret = np.zeros((DFF))
    for i in range(NU):
        for j in range(NTT):
            ret[i * NTT + j] = i;
    return ret


def get_data_Y1():
    ret = np.zeros((DF))
    for i in range(NU):
        for j in range(NT):
            ret[i * NT + j] = i;
    return ret

"""
train_X = get_train_X(SP)
train_y = get_train_Y()

test_X = get_test_X(SP2)
test_y = get_test_Y()

in_total = np.concatenate((train_X, test_X), axis=0)
out_total = np.concatenate((train_y, test_y), axis=0)

model = Sequential()
model.add(Dense(78, activation='sigmoid', input_shape=(NCC,)))
model.add(Dense(NU, activation='sigmoid'))

# sparse_categorical_crossentropy   mean_squared_error
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
# print (model.summary())

history = model.fit(in_total, out_total, shuffle=True, epochs=50, validation_split=0.1)

# history =model.fit(in_total,out_total,validation_split=0.1,shuffle=True,epochs=50)

pred_y = model.predict(test_X)  # test

fff = open("ann.txt", "a")
fff.write("==========================Shorter 3 s =======================================\n")
for ii in range(NU*NT):
    cd=np.zeros(shape=(NU))
    #C=model.predict(test_X[ii,:])
    C=pred_y[ii,:]
    cg = np.argmax(C)
    cd[cg]=1
    print(cd)
    fff.write(str(cd)+"\n")
    if((ii+1)%NT==0):
        fff.write("===================================================================================\n")
fff.close()
pred_y_train = model.predict(train_X)  # train
pred_y = np.argmax(pred_y, axis=1)
pred_y_train = np.argmax(pred_y_train, axis=1)  # train
print(pred_y)
# print(accuracy_score(train_y,pred_y))
print("confusion_matrix for training data : ")
# print(confusion_matrix(train_y,pred_y_train))
print("confusion_matrix for testing data : ")
print(confusion_matrix(get_data_Y1(), pred_y))
mat = confusion_matrix(get_data_Y1(), pred_y)
# print(classification_report(test_y,pred_y))

"""
    # Get training and test loss histories
    #training_loss = history.history['loss']
    #test_loss = history.history['val_loss']

    # Create count of the number of epochs
    #epoch_count = range(1, len(training_loss) + 1)
"""
labels = []
for i in range(NU):
    # print(i)
    #fname2 = imagePaths + 'test/speaker-' + str(i + 1) + '.wav'

    #for imagePath in imagePaths:
    label = 'speaker-' + str(i + 1)
    labels.append(label)
labels = np.array(labels)
model.save('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.model')
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("[INFO] serializing label binarizer...")
f = open('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

i = 0
fname2 = sdir + 'test/speaker-' + str(i + 1) + '.wav'
fname = sdir + 'train/speaker-' + str(i + 1) + '.wav'
# print(fname)
(fs, data) = wavfile.read(fname)
(fs2, data2) = wavfile.read(fname2)
# print(np.shape(data))
# data = data / np.max(np.abs(data))
# data2 = data2 / np.max(np.abs(data2))
mfcc1 = python_speech_features.base.mfcc(data, samplerate=fs, winlen=0.128, winstep=0.064, numcep=NC + 1,
                                            nfilt=NF,
                                            nfft=NP, preemph=0, appendEnergy=False, winfunc=np.hamming)

mfcc2 = python_speech_features.base.mfcc(data2, samplerate=fs2, winlen=0.128, winstep=0.064, numcep=NC + 1,
                                            nfilt=NF,
                                            nfft=NP, preemph=0, appendEnergy=False, winfunc=np.hamming)
mfcc = mfcc1[:, 1:]  # remove first column coeff 0

mfcc0 = mfcc2[:, 1:]  # remove first column coeff 0
delta = python_speech_features.base.delta(mfcc, 1)

delta2 = python_speech_features.base.delta(mfcc0, 1)
delta_delta = python_speech_features.base.delta(delta, 1)

delta2_delta = python_speech_features.base.delta(delta2, 1)
cc = np.concatenate((mfcc.T, delta.T, delta_delta.T), axis=0)

cc2 = np.concatenate((mfcc0.T, delta2.T, delta2_delta.T), axis=0)

temp = cc[:, 0:NTT]

temp2 = cc2[:, 0:NT]
SP[i, :, :] = temp
SP2 = np.zeros([NU, NCC, NT])
SP2[i, :, :] = temp2
def get_test_X1(SP):
    ret = np.zeros((NT, NCC))
    temp = SP.T
    for i in range(NT):
        ret [i] = temp[i];
    return ret

test_X = get_test_X1(temp2)
test_y = get_test_Y()

pred_y = model.predict(test_X)
print(np.argmax(pred_y,axis=1))
pp = np.bincount(np.argmax(pred_y,axis=1))
print(pp)

print(np.argmax(pp))
#print(ppp)
"""