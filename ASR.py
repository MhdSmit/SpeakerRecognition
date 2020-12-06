from scipy.io import wavfile
import numpy as np
import python_speech_features
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
import os
import pickle

np.random.seed(1)

def load_audio(path):
    """
    Load audio file
    :param path: path to audio file
    :return: sample rate, audio file read using scipy.io
    """
    try:
        (fs, data) = wavfile.read(path)
    except:
        print("File '%s' not found"%path)

    return fs, data


def extract_mfcc_features(data, samplerate, winlen=0.128, winstep=0.064, numcep= 14,
                          nfilt= 44, nfft=1024, appendEnergy=False, winfunc=np.hamming):
    mfcc_data = python_speech_features.base.mfcc(data, samplerate=samplerate, winlen=winlen, winstep=winstep, numcep=numcep,
                                            nfilt=nfilt, nfft=nfft, appendEnergy=appendEnergy, winfunc=winfunc)
    return mfcc_data[:,1:] #Remove Energy Component

def extract_delta (data):
    return python_speech_features.base.delta(data, 1)

def convert_to_one_hot(lst, spks_num):
    hot_one =[]
    for (spk_idx, rep)in lst:
        true_label = np.zeros((spks_num))
        true_label[spk_idx] = 1
        hot_one.extend([true_label]*rep)
    return np.asarray(hot_one)

def concatenate_list_of_arrays(lst):
    concated = lst[0] #first array
    for i in range(1,len(lst)):
        concated = np.concatenate((concated,lst[i]), axis=0)
    return concated

def prepare_train_val_data(speakers_dir, spks_num):
    """
    :param speakers_dir: a directory contains list of wav files, each one belongs to unique speaker
    :return: train_X, train_Y
    """
    trainX =[]
    trainY =[]
    spk_idxes = {}
    cur_idx = 0
    for spk in os.listdir(speakers_dir):
        if spk[-4:] == ".wav":
            # get speaker id
            spk_id = spk[:10] # we assume that first 10 chars represent speaker id
            #set new idx or get current idx
            if spk_id in spk_idxes.keys():
                spk_idx = spk_idxes[spk_id]
            else:
                spk_idx = cur_idx
                spk_idxes[spk_id] = spk_idx
                cur_idx = cur_idx + 1
            # read_file
            filename = os.path.join(speakers_dir, spk )
            sr, data = load_audio(filename)
            #extract mfcc, deltas, and delta-delats
            mfcc = extract_mfcc_features(data,sr)
            d_mfcc = extract_delta(mfcc)
            dd_mfcc = extract_delta(d_mfcc)
            features = np.concatenate((mfcc,d_mfcc,dd_mfcc), axis = 1)
            #add to trainX
            features = features[:,:]
            trainX.append(features)
            #expand train Y
            trainY.append((spk_idx,features.shape[0]))

    return concatenate_list_of_arrays(trainX), convert_to_one_hot(trainY,spks_num),spk_idxes

def extract_features(wav_path):
    # read_file
    sr, data = load_audio(wav_path)
    # extract mfcc, deltas, and delta-delats
    mfcc = extract_mfcc_features(data, sr)
    d_mfcc = extract_delta(mfcc)
    dd_mfcc = extract_delta(d_mfcc)
    features = np.concatenate((mfcc, d_mfcc, dd_mfcc), axis=1)
    return features


def get_ANN_model_1(input_dim, out_dim, hidden_dim = 78,optimizer = None, loss = None,metrics=None):
    """
    Create model 1
    :param input_dim:  Number of units in input layer (equal to number of feautures)
    :param out_dim: Number of units in output layer (equal to number of speakers)
    :param optimizer: optimizer name of type string or optimizer class
    :param loss: loss name
    :param metrics: metrics to calculate
    :return: compiled model
    """
    model = Sequential()
    model.add(Dense(hidden_dim, activation='sigmoid', input_shape=(input_dim,)))
    model.add(Dense(out_dim, activation='sigmoid'))

    # Init params
    if optimizer is None:
        optimizer='adam'
    if loss is None:
        loss = 'mean_squared_error'
        #loss = 'categorical_crossentropy'
    if metrics is None:
        metrics = ['acc']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_validate_model(model, train_dataX, train_dataY,test_dataX,test_dataY ,shuffle =True, epochs = 50):
    """
    Train and validate model
    :param model: complied model to train
    :param data_X:
    :param data_Y:
    :return: trained model
    """
    model.fit(train_dataX,train_dataY,shuffle=shuffle,epochs=epochs,validation_data=(test_dataX,test_dataY))
    return model

def train_val_pipeline(train_spks_dir,test_spks_dir, spks_num, model_hidden_dim,epochs, model_output_path):
    train_dataX, train_dataY, spk_dict = prepare_train_val_data(train_spks_dir,spks_num)
    test_dataX, test_dataY, spk_dict = prepare_train_val_data(test_spks_dir, spks_num)
    model = get_ANN_model_1(input_dim= train_dataX.shape[1],out_dim= spks_num, hidden_dim = model_hidden_dim)
    print(model.summary())
    model = train_validate_model(model,train_dataX,train_dataY,test_dataX,test_dataY,epochs=epochs)
    print("saving model")
    model.save(model_output_path)
    return model, spk_dict

def predict_speaker(wav_file):
    spk_dict_file = './spk_dict.pickle'
    print("1")
    print(wav_file)
    trained_model = load_model('./model.h5')
    print("1")
    test_dataX= extract_features(wav_file)
    print("1")
    x = trained_model.predict(test_dataX,batch_size=2)
    print("1")
    spk_idx= np.argmax(x[0])
    with open(spk_dict_file, 'rb') as handle:
        spk_dict = pickle.load(handle)

    for id, idx in spk_dict.items():
        if idx == spk_idx:
            return id
    return "Not Recognized"

if __name__ == '__main__':
    """"
    train_spks_dir = "./train"
    test_spks_dir = "./test"
    spk_dict_file = './spk_dict.pickle'
    spks_num = 11
    model_hidden_dim = 78
    model_output_path = "model.h5"
    epochs = 50
    model, spk_dict = train_val_pipeline(train_spks_dir,test_spks_dir, spks_num, model_hidden_dim,epochs =epochs, model_output_path = model_output_path)

    with open(spk_dict_file ,'wb') as handle:
        pickle.dump(spk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    """
    spk = predict_speaker('.//test//speaker-4.wav')
    print(spk)
    #"""
