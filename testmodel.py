from scipy.io import wavfile
import numpy as np
import python_speech_features
import pickle
from keras.models import load_model
np.random.seed(1)
# Sounds
NU = 10  # number of speaker
# MFCC parameters
NC = 13  # number of coeff
NF = 44  # number of filters
NP = 1024  # number of samples in a frame (128 ms)
INC = 512  # increment between frames
NCC = 3 * NC  # total number of coefficients
# Speakers cof
NT = 50  # number of frames for testing from each user
DF = NT * NU  # all data frames
NTT = 1900
DFF = NTT * NU
SP2 = np.zeros([NU, NCC, NT])  # same as previous but for test

def get_test_X1(SP):
    ret = np.zeros((NT, NCC))
    temp = SP.T
    for i in range(NT):
        ret[i] = temp[i]
    return ret

def get_test_Y():
    rett = []
    for i in range(NU):
        for j in range(NT):
            ret = np.zeros((NU))
            ret[i] = 1
            rett.append(ret)
    return np.asarray(rett)

model = load_model('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.model')
lb = pickle.loads(open('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.pickle', "rb").read())
labels = []
for i in range(NU):
    label = 'speaker-' + str(i + 1)
    labels.append(label)
labels = np.array(labels)
sdir ='C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ahlam data//test//'
fileName = 'speaker-9.wav'
fname2 = sdir + fileName
(fs2, data2) = wavfile.read(fname2)
mfcc2 = python_speech_features.base.mfcc(data2, samplerate=fs2, winlen=0.128, winstep=0.064, numcep=NC + 1,
                                            nfilt=NF,
                                            nfft=NP, preemph=0, appendEnergy=False, winfunc=np.hamming)
mfcc0 = mfcc2[:, 1:]  # remove first column coeff 0
delta2 = python_speech_features.base.delta(mfcc0, 1)
delta2_delta = python_speech_features.base.delta(delta2, 1)
cc2 = np.concatenate((mfcc0.T, delta2.T, delta2_delta.T), axis=0)
temp2 = cc2[:, 0:NT]
test_X = get_test_X1(temp2)
test_y = get_test_Y()
pred_y = model.predict(test_X)
print(np.argmax(pred_y,axis=1))
pp = np.bincount(np.argmax(pred_y,axis=1))
print(pp)
print(np.argmax(pp))
labels = ['speaker-1', 'speaker-2', 'speaker-3', 'speaker-4', 'speaker-5', 'speaker-6', 'speaker-7', 'speaker-8',
          'Bashar', 'speaker-10']
print('the result is')
print(labels[np.argmax(pp)])