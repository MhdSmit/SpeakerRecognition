from tkinter import ttk
from tkinter import *
import tkinter as tk
from tkinter import messagebox
#import speech_recognition as sr
from pygame import mixer
import pygame as pg
import time
import os
import pyaudio
import wave
import threading
from ASR import predict_speaker
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
# Recording vars
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 8000  # Record at 8000 samples per second
filename = "recorded_audio.wav"
p = pyaudio.PyAudio()
is_recording = False
##########
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

###############
def save_wav(frames):
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Audio Saved to " + filename)
model = load_model('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.model')
lb = pickle.loads(open('C://Users//mohammad//Documents//Python Scripts//SpeakerRecognition//ANNv7.pickle', "rb").read())
labels = []
for i in range(NU):
    label = 'speaker-' + str(i + 1)
    labels.append(label)
labels = np.array(labels)
######## RECORING THREAD
class RecordingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        frames = []
        global is_recording
        while (is_recording):
            print("RECORDING ...")
            stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)

            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        save_wav(frames)

r = RecordingThread() #global var
######## GUI config
root = tk.Tk()
root.title('Recorder')
root.iconbitmap('mic.ico')

col_count = 25
row_count = 15
for col in range(col_count):
    root.grid_columnconfigure(col, minsize=20)

for row in range(row_count):
    root.grid_rowconfigure(row, minsize=20)
####

photo = PhotoImage(file='microphone.png').subsample(35, 35)

status_lbl = ttk.Label(root, text='Click on Mic to start recording')
status_lbl.grid(row=5, column=0, columnspan=25)

predict_spk_label = ttk.Label(root, text='')
predict_spk_label.grid(row=11, column=0, columnspan=25)



def record_btn_clicked():
    mixer.init()
    mixer.music.load('chime1.mp3')
    mixer.music.play()
    #set labels
    status_lbl['text'] = "Recording..."
    predict_spk_label['text'] = ""

    #global vars
    global r
    global is_recording

    is_recording = True
    r = RecordingThread()
    r.start()


def stop_recording_clicked():
    global r
    global is_recording
    is_recording = False
    r.join()
    status_lbl['text'] = "Click 'Predict' to get speaker name, or click on Mic to start recording agian"

def play():

    f = wave.open("recorded_audio.wav", "rb")
    # instantiate PyAudio
    player = pyaudio.PyAudio()
    # open stream
    stream = player.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

        # stop stream
    stream.stop_stream()
    stream.close()
    player.terminate()


def play_recording():
    status_lbl['text'] = "Playing..."
    root.update()

    th = threading.Thread(target=play)
    th.start()

    th.join()
    status_lbl['text'] = "Click 'Predict' to get speaker name, or click on Mic to start recording agian"

def predict_rec_clicked():

    fname2 = filename
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
    #print(np.argmax(pred_y, axis=1))
    pp = np.bincount(np.argmax(pred_y, axis=1))
    #print(pp)
    #print(np.argmax(pp))
    labels = ['speaker-1', 'speaker-2', 'speaker-3', 'speaker-4', 'speaker-5', 'speaker-6', 'speaker-7', 'speaker-8',
              'speaker-9', 'speaker-10']
    #print('the result is')
    #print(labels[np.argmax(pp)])
    predict_spk_label['text'] = "predicted speaker: " + labels[np.argmax(pp)]


rec_btn = ttk.Button(root, image=photo, command=record_btn_clicked)#, activebackground='#c1bfbf', overrelief='groove', relief='sunken')
rec_btn.grid(row=2, column=1, padx=(10, 10))

stop_rec = ttk.Button(root, text='Stop Recording', width=20, command=stop_recording_clicked)
stop_rec.grid(row=2, column=4,columnspan=3)

play_rec = ttk.Button(root, text='Play Recording', width=20, command=play_recording)
play_rec.grid(row=2, column=18,columnspan=3)

predict_rec = ttk.Button(root, text='Predict Speaker', width=20, command=predict_rec_clicked)
predict_rec.grid(row=8, column=11,columnspan = 2)

root.wm_attributes('-topmost', 1)
root.mainloop()