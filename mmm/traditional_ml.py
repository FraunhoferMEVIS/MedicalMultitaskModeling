from typing import cast
import numpy as np
import wandb
import logging
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
from scipy.stats import kendalltau


def evaluate_on_clf_arrays(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names=None,
    confmattitle=None,
    confmatonlyval=True,
):
    sklearn_compatibles = {
        "randomforest": RandomForestClassifier,
    }
    softmax = torch.nn.Softmax(dim=1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    imp = None
    for clf_name, clf_type in sklearn_compatibles.items():
        clf = clf_type()
        clf = clf.fit(X_train, y_train)
        y_pred_train: np.ndarray = clf.predict(X_train)
        y_pred_val: np.ndarray = clf.predict(X_val)
        if clf_name == "svm":
            decision = torch.from_numpy(cast(SVC, clf).decision_function(X_val))
            # logging.info(f"{decision.shape}")
            if len(decision.shape) == 1:
                decision = decision.unsqueeze(0)
            y_pred_proba: np.ndarray = softmax(decision).squeeze().numpy()
        else:
            y_pred_proba: np.ndarray = clf.predict_proba(X_val)
            if y_pred_proba.shape[-1] == 2:
                y_pred_proba = y_pred_proba[:, -1]
        res = {
            f"{clf_name}_train_acc": accuracy_score(y_true=y_train, y_pred=y_pred_train),
            # f"{clf_name}_train_confmatarr": confusion_matrix(y_true=y_train, y_pred=y_pred_train),
            f"{clf_name}_val_acc": accuracy_score(y_true=y_val, y_pred=y_pred_val),
            # f"{clf_name}_val_confmatarr": confusion_matrix(y_true=y_val, y_pred=y_pred_val),
            # f"{clf_name}_val_roc_auc": roc_auc_score(y_true=y_val, y_score=y_pred_proba, multi_class="ovr"),
            f"{clf_name}_val_f1": f1_score(y_true=y_val, y_pred=y_pred_val, average="micro"),
            # f"{clf_name}_val_f1_macro": f1_score(y_true=y_val, y_pred=y_pred_val, average="macro"),
            # f"{clf_name}_val_f1_weighted": f1_score(y_true=y_val, y_pred=y_pred_val, average="weighted"),
        }
        if clf_name == "randomforest":
            imp = clf.feature_importances_
        if confmattitle is not None:
            logging.info(f"{confmattitle}_{clf_name}_val")
            if not confmatonlyval:
                res[f"{clf_name}_train_confmat"] = wandb.plot.confusion_matrix(
                    preds=list(y_pred_train.tolist()),
                    y_true=list(y_train.tolist()),
                    class_names=class_names,
                    title=f"{confmattitle}_{clf_name}_train",
                )
            res[f"{clf_name}_val_confmat"] = wandb.plot.confusion_matrix(
                preds=list(y_pred_val.tolist()),
                y_true=list(y_val.tolist()),
                class_names=class_names,
                title=f"{confmattitle}_{clf_name}_val",
            )

        yield res, imp


def evaluate_on_reg_arrays(X_train, y_train, X_val, y_val, class_names=None):
    sklearn_compatibles = {
        "randomforest": RandomForestRegressor,
    }
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    for clf_name, clf_type in sklearn_compatibles.items():
        clf = clf_type()
        clf = clf.fit(X_train, y_train)
        y_pred_train: np.ndarray = clf.predict(X_train)
        y_pred_val: np.ndarray = clf.predict(X_val)
        # y_proba_train: np.ndarray = clf.predict_proba(X_train)
        # y_proba_val: np.ndarray = clf.predict_proba(X_val)
        res = {
            # f"{clf_name}_train_kendalltau": kendalltau(x=y_train, y=y_pred_train),
            f"{clf_name}_train_mse": mean_squared_error(y_true=y_train, y_pred=y_pred_train),
            f"{clf_name}_val_mse": mean_squared_error(y_true=y_val, y_pred=y_pred_val),
            # f"{clf_name}_val_auc": roc_auc_score(y_true=y_val, y_score=y_pred_val)
        }
        imp = None
        if clf_name == "randomforest":
            imp = clf.feature_importances_

        yield res, imp
