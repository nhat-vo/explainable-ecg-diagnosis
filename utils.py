import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from tqdm import tqdm

from typing import List, Set, Optional

import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS_FULL = [
    "1st degree AV block (1dAVb)",
    "right bundle branch block (RBBB)",
    "left bundle branch block (LBBB)",
    "sinus bradycardia (SB)",
    "atrial fibrillation (AF)",
    "sinus tachycardia (ST)",
]
LABEL_FILTERS = [
    {'bloc A-V du premier'},
    {'Bloc de branche droit complet', 'BBD'},
    {'Bloc de branche gauche complet', 'BBG'},
    {'Bradycardie sinusale'},
    {'Fibrillation auriculaire'},
    {'Tachycardie sinusale'},
]
LABELS = [
    '1dAVb',
    'RBBB',
    'LBBB',
    'SB',
    'AF',
    'ST',
]


def plot_ecg(X: np.array, titles:Optional[List[str]]=None) -> None:
    titles = titles if titles else LEADS
    for c in range(12):
        plt.figure(figsize=(20, 5))
        plt.plot(X[:, c])
        plt.title(titles[c])
        plt.grid()

    plt.show()


def get_diagnosis(filename: str) -> str:
    root = ET.parse(filename).getroot()
    node = root[6]
    
    diagnosis = 'Diagnosis:\n'
    for i in node:
        if i.tag == "DiagnosisStatement":
            for j in i:
                if j.tag == "StmtText":
                    diagnosis += '\n' + j.text
    return diagnosis


def read_hcu_data(verbose=True):
    files = []
    with open('data/valid.txt') as file_list:
        for line in file_list:
            files.append(line.strip().split('.xml')[0])
    ecg = []
    diagnosis = []
    
    files = tqdm(files) if verbose else files
    for file in files:
        data = pd.read_csv('data/csv/' + file.split('/')[-1] + '.csv').drop(columns=["Unnamed: 12", "Unnamed: 13", "V4R"], errors='ignore')
        diag = get_diagnosis(os.path.join('data', file + '.xml'))
        ecg.append(data)
        diagnosis.append(diag)
    ecg = np.array(ecg)
    return ecg, diagnosis


def extract_ground_truth(diagnosis: List[str], filters:List[Set[str]]=None, verbose=True) -> np.array:
    filters = filters if filters else LABEL_FILTERS
    Y_ground = []
    diagnosis = tqdm(diagnosis) if verbose else diagnosis
    for diag in diagnosis:
        y = []
        for group in filters:
            for f in group:
                if f.lower() in diag.lower():
                    y.append(True)
                    break
            else:
                y.append(False)
        Y_ground.append(y)
    return np.array(Y_ground)



def load_ribeiro_model(path:Optional[str]=None):
    path = path if path else 'model/model.hdf5'
    model = load_model("model/model.hdf5", compile=False)
    model.compile(loss="binary_crossentropy", optimizer=Adam())
    return model


def preprocess_ecg(ecg, ampl_ratio=1/500, ori_freq=500, tar_freq=400, tar_len=4096):
    if len(ecg.shape) == 2:
        X_ecg = ecg[None, ...]
    else:
        X_ecg = np.array(ecg)
    assert len(X_ecg.shape) == 3, "ECG signals must be of shape (num_sample x signal_length x num_leads)"
    
    X_ecg = X_ecg * ampl_ratio
    
    # padding / trimming
    length = X_ecg.shape[1] * tar_freq // ori_freq
    X_ecg = scipy.signal.resample(X_ecg, length, axis=1)
    if length > tar_len:
        left = (length - tar_len) // 2
        right = length - tar_len - left
        X_ecg =  X_ecg[:, left:-right]
    elif length < tar_len:
        left = (tar_len - length) // 2
        right = tar_len - length - left
        X_ecg = np.pad(X_ecg, ((0, 0), (left, right), (0, 0)), "constant")
    return X_ecg



def fit_adaptors(X, Y, init_fn, **kwargs):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    adaptors = []
    for i in range(6):
        adaptor = init_fn(**kwargs)
        adaptor.fit(X, Y[:, i])
        adaptors.append(adaptor)
    return adaptors



















