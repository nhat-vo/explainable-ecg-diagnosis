#!/usr/bin/env python
# coding: utf-8

# # Smooth Grad

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import neurokit2 as nk
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Helper function and constants

# In[3]:


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DIAGS = [
    "1st degree AV block (1dAVb)",
    "right bundle branch block (RBBB)",
    "left bundle branch block (LBBB)",
    "sinus bradycardia (SB)",
    "atrial fibrillation (AF)",
    "sinus tachycardia (ST)",
]


# In[4]:


def plot_ecg(X):
    for c in range(12):
        plt.figure(figsize=(20, 5))
        plt.plot(X[:, c])
        plt.title(LEADS[c])
        plt.grid()

    plt.show()


# ## Extract data

# In[5]:


import xml.etree.ElementTree as ET


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


# In[6]:


import os
files = []
with open('data/valid.txt') as file_list:
    for line in file_list:
        files.append(line.strip().split('.xml')[0])


# In[7]:


ecg = []
diagnosis = []
for file in files:
    data = pd.read_csv('data/csv/' + file.split('/')[-1] + '.csv').drop(columns=["Unnamed: 12", "Unnamed: 13", "V4R"], errors='ignore')
    diag = get_diagnosis(os.path.join('data', file + '.xml'))
    ecg.append(data)
    diagnosis.append(diag)
    # print(file, '\n', diagnosis, '\n', '--------------------------', '\n')
ecg = np.array(ecg)


# In[8]:


# plot_ecg(X[0])


# In[9]:


filters = [
    {'bloc A-V du premier'},
    {'Bloc de branche droit complet', 'BBD'},
    {'Bloc de branche gauche complet', 'BBG'},
    {'Bradycardie sinusale'},
    {'Fibrillation auriculaire'},
    {'Tachycardie sinusale'},
]
    


# In[10]:


def ground_truth(diagnosis):
    Y_ground = []
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


# In[11]:


Y_ground = ground_truth(diagnosis)


# ## Load Model

# In[12]:


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

model = load_model("model/model.hdf5", compile=False)
model.compile(loss="binary_crossentropy", optimizer=Adam())


# ## Reformat data for model

# In[13]:


X_ecg = scipy.signal.resample(ecg, 4000, axis=-2)
if len(X_ecg.shape) == 2:
    X_ecg = X_ecg[None, ...]
X_ecg = np.pad(X_ecg, ((0, 0), (48, 48), (0, 0)), "constant")
X_ecg = X_ecg / 500
print(X_ecg.shape)


# In[14]:


X_ecg_inv =  []
for ecg in tqdm(X_ecg):
    ecg_inv = []
    # print(ecg.shape)
    for channel in range(12):
        ecg_inv.append(nk.ecg_invert(ecg[:, channel], sampling_rate=400)[0])
    X_ecg_inv.append(np.array(ecg_inv).T)
X_ecg_inv = np.array(X_ecg_inv)


# In[ ]:


X_ecg_inv.shape


# In[31]:


Y_ecg = model.predict(X_ecg)


# In[32]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# In[33]:


for i in range(6):
    print(DIAGS[i], f1_score(Y_ground[:, i], Y_ecg[:, i] > 0.5))


# In[35]:


def fit_dtrees(X, Y):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    trees = []
    for i, label in enumerate(DIAGS):
        dtree = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        dtree.fit(X, Y[:, i])
        trees.append(dtree)
    return trees


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(Y_ecg, Y_ground, test_size=0.2)


# In[37]:


dtrees = fit_dtrees(X_train, Y_train)


# In[180]:


for i in range(6):
    dtree = RandomForestClassifier(n_estimators=100,  class_weight='balanced')
    print(DIAGS[i], cross_val_score(dtree, X_train, Y_train[:, i], scoring='f1'))


# In[181]:


for i in range(6):
    print(DIAGS[i], f1_score(Y_test[:, i], dtrees[i].predict(X_test)))


# In[69]:


dtrees[3].score(X_test, Y_test[:, 3])


# In[ ]:





# In[110]:


Y_train.shape


# In[152]:


adapter = SVC(C=1e4, probability=True, class_weight='balanced')
adapter.fit(X_train, Y_train[:, 0])


# In[153]:


pred = adapter.predict(X_test)


# In[154]:


f1_score(Y_test[:, 0], pred)


# In[155]:


adapter.score(X_test, Y_test[:, 0])


# In[156]:


f1_score(Y_test[:, 0], X_test[:, 0] > 0.5)


# ## SmoothGrad

# In[13]:


from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore


# In[14]:


saliency = Saliency(model, clone=True)
def score_function(output, i):
    return (output[0][i],)

def visualize_smoothgrad(X):
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))
    for i in range(6):
        # Generate saliency map
        saliency_map = saliency(
            lambda x: score_function(x, i),
            X,
            keepdims=True,
            smooth_samples=10,
            smooth_noise=0.01,
            normalize_map=False
        )
        print(np.max(saliency_map))
        ax[i].set_title(DIAGS[i], fontsize=16)
        ax[i].imshow(np.repeat(saliency_map[0], 100, axis=-1).T, vmax=0.01, cmap='Oranges', alpha=0.8)
        ax[i].set_yticks(ticks=[i * 100 + 50 for i in range(12)], labels=LEADS)
        
        for c in range(12):
            scale = 45 / np.max(np.abs(X[:, c]))
            ax[i].plot(X[:, c] * scale + 50 + 100 * c)
    plt.show()


# In[16]:


visualize_smoothgrad(X[195])


# ## GradCam++

# In[15]:


from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus


# In[16]:


gradcam = GradcamPlusPlus(model, clone=True)
def score_function(output, i):
    return (output[0][i],)

def visualize_gradcam(X):
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))
    for i in range(6):
        # Generate saliency map
        gradcam_map = gradcam(
            lambda x: score_function(x, i),
            X,
            normalize_cam=False
        )
        print(np.max(gradcam_map))
        ax[i].set_title(DIAGS[i], fontsize=16)
        ax[i].imshow(np.repeat(gradcam_map[0], 100, axis=-1).T, vmax=0.05, cmap='Oranges')
        ax[i].set_yticks(ticks=[i * 100 + 50 for i in range(12)], labels=LEADS)
        
        for c in range(12):
            scale = 45 / np.max(np.abs(X[:, c]))
            ax[i].plot(X[:, c] * scale + 50 + 100 * c)
    plt.show()


# In[17]:


visualize_gradcam(X[7])


# ## ScoreCam

# In[18]:


from tf_keras_vis.scorecam import Scorecam


# In[19]:


scorecam = Scorecam(model, clone=True)
def score_function(output, i):
    return (output[0][i],)

def visualize_scorecam(X):
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))
    for i in range(6):
        # Generate saliency map
        scorecam_map = scorecam(
            lambda x: score_function(x, i),
            X,
            normalize_cam=False
        )
        print(np.max(scorecam_map))
        ax[i].set_title(DIAGS[i], fontsize=16)
        ax[i].imshow(np.(scorecam_map[0], (4096, 1200)).T, vmax=0.05, cmap='Oranges')
        ax[i].set_yticks(ticks=[i * 100 + 50 for i in range(12)], labels=LEADS)
        
        for c in range(12):
            scale = 45 / np.max(np.abs(X[:, c]))
            ax[i].plot(X[:, c] * scale + 50 + 100 * c)
    plt.show()


# In[12]:


# visualize_scorecam(X[4])


# ## LayerCam

# In[21]:


from tf_keras_vis.layercam import Layercam


# In[22]:


layercam = Layercam(model, clone=True)
def score_function(output, i):
    return (output[0][i],)

def visualize_layercam(X):
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))
    for i in range(6):
        # Generate saliency map
        layercam_map = layercam(
            lambda x: score_function(x, i),
            X,
            normalize_cam=False
        )
        print(np.max(layercam_map))
        ax[i].set_title(DIAGS[i], fontsize=16)
        ax[i].imshow(resize(layercam_map[0], (4096, 1200)).T, vmax=0.05, cmap='Oranges')
        ax[i].set_yticks(ticks=[i * 100 + 50 for i in range(12)], labels=LEADS)
        
        for c in range(12):
            scale = 45 / np.max(np.abs(X[:, c]))
            ax[i].plot(X[:, c] * scale + 50 + 100 * c)
    plt.show()


# In[23]:


visualize_layercam(X[7])


# ## LIME

# In[134]:


from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb


# In[111]:


def visualize_explanation(X, explain_func, lead=None):
    exps, pred, vmin, vmax, cmap = explain_func(X)
    print(vmin, vmax)
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))

    for label in range(6):
        # Generate saliency map
        exp = exps[label]
        
        
        if lead is None:
            img = np.swapaxes(np.repeat(exp, 100, axis=1), 0, 1)
            print(img.shape)
        else:
            exp0 = exp
            exp = exp[:, [lead], ...]
            print(exp.shape)
            img = np.swapaxes(np.repeat(exp, 1200, axis=1), 0, 1)
            print(img.shape)
            # return exp0, exp, img
        ax[label].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        
        ax[label].set_title(DIAGS[label] + " " + str(pred[0, label]) , fontsize=16)
        
        xticks = np.arange(0, 4096, 400)
        ax[label].set_xticks(ticks=xticks, labels=xticks // 400)
        ax[label].set_xlabel('time (s)')
        ax[label].set_ylabel('lead')
        ax[label].grid()
        
        
        if lead is None:
            for c in range(12):
                scale = -45 / np.max(np.abs(X[:, c]))
                ax[label].plot(X[:, c] * scale + 50 + 100 * c)
            ax[label].set_yticks(ticks=[50 + j * 100 for j in range(12)], labels=LEADS)
        else:
            ax[label].set_ylabel(LEADS[lead])
            scale = -550 / np.max(np.abs(X[:, lead]))
            ax[label].plot(X[:, lead] * scale + 600)
            ax[label].set_yticks([])
    
    plt.show()
    return exps


# In[135]:


def visualize_lime(X, lead=None):
    def predict_img(X):
        # X_ecg = model.predict(X[..., 0], verbose=False)
        X_ecg = model.predict(rgb2gray(X), verbose=False)
        # return X_ecg
        # print(X_ecg.shape)
        res = np.array([dtree.predict_proba(X_ecg) for dtree in dtrees])
        res = res[...,1]
        return res.T
        
    def exp_func(X):
        # X_img = np.stack((X,) * 3, axis=-1)
        X_img = gray2rgb(X)
        explainer = lime_image.LimeImageExplainer()
        exp = explainer.explain_instance(X_img, predict_img, top_labels=6)
        labels = exp.local_exp.keys()
        pred = predict_img(np.array([X_img]))
        
        # print(pred.shape)

        exps = []
        for label in range(6):
            temp, mask = exp.get_image_and_mask(label, positive_only=False, hide_rest=True)
            # temp /= temp.max()
            temp = np.where(temp == 0, 1, temp)
            temp = temp / 2 + 0.5
            exps.append(temp)
        return exps, pred, None, None, None
    visualize_explanation(X, exp_func, lead=lead)


# In[136]:


visualize_lime(X_ecg[803])


# In[86]:


Y_ecg[195]


# In[ ]:


dtrees[3].predict([Y_ecg[195]])


# In[17]:


Y[50]


# In[ ]:





# ## Shap

# In[129]:


import shap
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


# In[137]:


def visualize_shap(X, lead=None):
    def predict_img(X):
        X_ecg = model.predict(X[..., 0], verbose=False)
        # return X_ecg
        res = np.array([dtree.predict_proba(X_ecg) for dtree in dtrees])
        res = res[...,1]
        return res.T
    
    def explain_func(X):
        X_img = np.stack((X,) * 3, axis=-1)
        if len(X_img.shape) == 3:
            X_img = X_img[None, ...]

        masker = shap.maskers.Image("blur(128,128)", X_img[0].shape)
        explainer = shap.Explainer(predict_img, masker, output_names=DIAGS)

        shap_values = explainer(X_img, max_evals=5000, batch_size=50)
        labels = shap_values.output_names
        print(labels)
    
        norm = np.max(np.abs(shap_values.values[0, ...]))
        norm = max(norm, 1e-5)
        
        pred = predict_img(X_img)
        
        print(shap_values.values.shape)
        print(np.min(sha_values.values[0, ...]))
        
        exps = [shap_values.values[0,..., 0, i] for i in range(6)]
        
        return exps, pred, norm
    
    visualize_explanation(X, explain_func, lead=lead)



# In[127]:


# def predict_img(X):
#     return model.predict(X[..., 0], verbose=False)
    # return model.predict(X, verbose=False)

def visualize_shap_deep(X):
    # X_img = np.stack((X,) * 3, axis=-1)
    X_img = X
    if len(X_img.shape) == 2:
        X_img = X_img[None, ...]
    # masker = shap.maskers.Image("blur(128,128)", X_img[0].shape)
    explainer = shap.DeepExplainer(model, X_ecg[np.random.choice(X_ecg.shape[0], 100, replace=False)])
    shap_values = explainer.shap_values(X_img)
    # shap_values = explainer(X_img, max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip)
    # labels = shap_values.output_names
    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 36))
    
    norm = np.max(np.abs(shap_values))
    print(norm)

    for i, label in enumerate(DIAGS):
        img = np.swapaxes(np.repeat(shap_values[i][0] , 100, axis=-1), 0, 1)
        ax[i].imshow(img, extent=(0, 4096, 0, 1200), cmap='Oranges', vmax = norm)
        
        ax[i].set_title(label, fontsize=16)
        
        xticks = np.arange(0, 4096, 400)
        ax[i].set_xticks(ticks=xticks, labels=xticks // 400)
        ax[i].set_yticks(ticks=[j * 100 + 50 for j in range(12)], labels=LEADS)
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel('lead')
        
        for c in range(12):
            scale = 45 / np.max(np.abs(X[:, c]))
            ax[i].plot(X[:, c] * scale + 50 + 100 * c)
        ax[i].grid()
    plt.show()


# In[41]:


np.random.choice(np.arange(len(X_ecg))[Y_ground[:, 2] == 1])


# In[42]:


Y_ecg[803]


# In[235]:


Y_ground[50]


# In[ ]:


visualize_shap(X_ecg[50])


# In[236]:


visualize_shap(X_ecg[50], 11)


# In[160]:


visualize_shap(X_ecg[50])


# In[237]:


visualize_shap(X_ecg[803])


# In[130]:


visualize_shap_deep(X_ecg[50])

