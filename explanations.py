from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
from lime import lime_image
from skimage.color import gray2rgb

from utils import LABELS, LEADS

shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

Explanation = partial(defaultdict, lambda: None)




def extract_lime_explanation(X, predict_fn):
    """
    Extract explanations of predict_fn on the instance X using LIME.
    Returns an Explanation object to be used in visualize_explanation().
    """
    X_img = gray2rgb(X)
    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(X_img, predict_fn, top_labels=6)
    pred = predict_fn(np.array([X_img]))

    exps = []
    for label in range(6):
        temp, _ = exp.get_image_and_mask(label, positive_only=False, hide_rest=True)
        # temp /= temp.max()
        temp = np.where(temp == 0, 1, temp)
        temp = temp / 2 + 0.5
        exps.append(temp)
    return {"ecg": X, "explanations": exps, "predictions": pred}


def extract_shap_partition_explanation(X, predict_fn):
    """
    Extract explanations of predict_fn on the instance X using SHAP PartitionExplainer.
    Returns an Explanation object to be used in visualize_explanation().
    """
    X_img = np.stack((X,) * 3, axis=-1)
    if len(X_img.shape) == 3:
        X_img = X_img[None, ...]

    masker = shap.maskers.Image("blur(128,128)", X_img[0].shape)
    explainer = shap.Explainer(predict_fn, masker)

    shap_values = explainer(X_img, max_evals=5000, batch_size=50)

    norm = np.max(np.abs(shap_values.values[0, ...]))
    norm = max(norm, 1e-5)

    pred = predict_fn(X_img)

    exps = [shap_values.values[0, ..., 0, i] for i in range(6)]
    return {"ecg": X, "explanations": exps, "predictions": pred, "vmax": norm}


def extract_shap_deep_explanation(X, model, background):
    """
    Extract explanations of predict_fn on the instance X using SHAP DeepExplainer.
    Returns an Explanation object to be used in visualize_explanation().
    """
    if len(X.shape) == 2:
        X = X[None, ...]
    explainer = shap.DeepExplainer(model, background)
    pred = np.array(model(X))
    shap_values = explainer.shap_values(X)

    norm = np.max(np.abs(shap_values))
    exps = []

    for i in range(6):
        exps.append(shap_values[i][0])

    return {
        "ecg": X[0],
        "explanations": exps,
        "predictions": pred,
        "vmax": norm,
        "cmap": "Oranges",
    }


def visualize_explanation(
    ecg, explanations, predictions, vmin=None, vmax=None, cmap=None, lead=None, labels=None
):
    """
    Visualize the explanations object. This function could be called as follows:
        ```
        expl = extract_lime_explanation(X, predict_fn)
        visualize_explanations(**expl)
        ```
    Params:
        ecg: the 12-lead ecg signal
        explanations: the explanation weights of the ECG signals
        predictions: the predictions/activations of the model on the ECG signals
        vmin, vmax, cmap: parameters to parse to plt.imshow() 
        lead: the lead to plot, None to plot every leads
        labels: the lists containing labels to plot, None to plot every labels
    """
    
    if labels is None:
        labels = range(6)
    
    if isinstance(labels, int):
        labels = [labels]
        
    rows = len(labels)
    _, ax = plt.subplots(nrows=rows, ncols=1, figsize=(20, 6 * rows))
    if rows == 1:
        ax = [ax]

    for row, label in enumerate(labels):
        # Generate saliency map
        exp = explanations[label]

        if lead is None:
            img = np.swapaxes(np.repeat(exp, 100, axis=1), 0, 1)
        else:
            exp = exp[:, [lead], ...]
            img = np.swapaxes(np.repeat(exp, 1200, axis=1), 0, 1)
        ax[row].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

        ax[row].set_title(
            f'{LABELS[label]} ($\lambda$={str(predictions[0, label])})', fontsize=16
        )

        xticks = np.arange(0, 4096, 400)
        ax[row].set_xticks(ticks=xticks, labels=xticks // 400)
        ax[row].set_xlabel("time (s)")
        ax[row].set_ylabel("lead")
        ax[row].grid()

        if lead is None:
            for c in range(12):
                assert np.max(np.abs(ecg[:, c])) > 0.0001, c
                scale = -45 / np.max(np.abs(ecg[:, c]))
                ax[row].plot(ecg[:, c] * scale + 50 + 100 * c)
            ax[row].set_yticks(ticks=[50 + j * 100 for j in range(12)], labels=LEADS)
        else:
            ax[row].set_ylabel(LEADS[lead])
            scale = -100 / np.max(np.abs(ecg[:, lead]))
            ax[row].plot(ecg[:, lead] * scale + 600)
            ax[row].set_yticks([600], labels=[0])

    plt.show()
