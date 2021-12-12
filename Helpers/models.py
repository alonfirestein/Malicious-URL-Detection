import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from abc import ABC
import seaborn as sns
from contextlib import redirect_stdout
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from Helpers.Constants import *
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


class Model(ABC):
    def train(self, x_train, y_train, **kwargs):
        """
        don't forget to remove the labels
        :param x_train: df | numpy | tensor with the features
        :param y_train: the labels ground truth
        :return: the trained model
        """
        raise NotImplementedError

    def predict(self, x_test) -> np.ndarray:
        """
        predict x_test to it's labels
        :param x_test: df | numpy | tensor with the features
        :return: numpy array with the
        """
        raise NotImplementedError

    def pipeline(self, x_train, y_train, x_test, y_test, model_name: str, do_cross_validation=True,
                 cross_validator=KFold(n_splits=5), **kwargs):
        """
        generic pipeline that export the model results inside folder called by the model_name the execution datetime

        :param x_train: numpy matrix features for train
        :param y_train: numpy labels vector for train
        :param x_test: numpy matrix features for test/validation
        :param y_test: numpy labels vector for test/validation
        :param model_name: descriptive name of the model and the experiment
        :param do_cross_validation: if you want to do a cross validation - may take some time
        :param cross_validator: the cross validation technique (KFold,ShuffleSplit etc.)
        :param kwargs: additional model arguments(like batch_size etc.)
        :return: None
        """
        sns.set_theme()
        dir_name = fr"{model_name}-{datetime.datetime.now().strftime('%m-%d--%H-%M-%S')}"
        dir_name = os.path.join(REPORT_DIRECTORY_NAME, dir_name)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, "model_configuration.json"), "w") as file:
            json.dump(kwargs, file, indent=4, sort_keys=True)
        with open(os.path.join(dir_name, "model_output.txt"), 'w') as f:
            with redirect_stdout(f):
                self.train(x_train, y_train, **kwargs)
                prediction = self.predict(x_test)
                enc = LabelEncoder()
                np.savetxt(os.path.join(dir_name, 'predictions.txt'), enc.fit_transform(prediction), fmt='%2d')
        class_report = classification_report(y_test, prediction)
        print(class_report)
        with open(os.path.join(dir_name, "classification_report.txt"), "w") as file:
            file.write(class_report)
        mat = confusion_matrix(y_test, prediction)
        plt.figure(figsize=(10, 7))
        sns.heatmap(mat, annot=True)
        plt.savefig(os.path.join(dir_name, "confusion_matrix.png"))
        plt.draw()
        loss_values = self.get_loss()
        if loss_values is not None:
            plt.plot(loss_values)
            plt.title("Loss Over Epochs")
            plt.draw()
            plt.savefig(os.path.join(dir_name, "loss.png"))
        if do_cross_validation:
            with open(os.path.join(dir_name, f"classification_report_cross_validation.txt"), "w") as file:
                for i, (train, test) in enumerate(list(cross_validator.split(x_train))):
                    curr_x_train, curr_x_test, curr_y_train, curr_y_test = x_train[train], x_train[test], y_train[
                        train], y_train[test]
                    self.train(curr_x_train, curr_y_train)
                    prediction = self.predict(curr_x_test)
                    file.write(f"{'-' * 25}{i + 1}{'-' * 25}\n{classification_report(curr_y_test, prediction)}\n")
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            plt.bar(list(feature_importance.index), feature_importance.values)
            plt.title("Feature Importance")
            plt.draw()
            plt.savefig(os.path.join(dir_name, "feature_importance.png"))

    def get_loss(self) -> np.ndarray:
        """
        can be implemented to if you want the pipeline to export it
        :return: the loss array over the train iteration
        """
        pass

    def get_feature_importance(self) -> pd.Series:
        """
        return: value counts like series with feature names and their importance weights
        """
        pass

    def try_different_confidences(self, x_test, y_test, confidences=(0.5, 0.6, 0.7, 0.8)):
        pred = self.predict(x_test)
        for thr in confidences:
            new_pred = pred.copy()
            new_pred[pred > thr] = 1
            new_pred[pred <= thr] = 0
            print(classification_report(y_test, new_pred))


class SimpleDecisionTreeClassifier(Model):
    def train(self, x_train, y_train, **kwargs):
        self.decsion_tree_model = DecisionTreeClassifier(random_state=42, )
        self.decsion_tree_model.fit(x_train, y_train, **kwargs)

    def test(self, x_test):
        return self.decsion_tree_model.predict(x_test)


class XgboostModelBinary(Model):
    def train(self, x_train, y_train, **kwargs):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        param = {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "gamma": 0, "subsample": 0.7,
                 "objective": 'binary:logistic', "scale_pos_weight": 1, "seed": 93, "eval_metric": "logloss"}
        # "colsample_bytree":0.8# evals=evals, early_stopping_rounds=10
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_eval = xgb.DMatrix(x_val, label=y_val)
        evallist = [(d_train, 'train'), (d_eval, 'eval')]
        self.bst = xgb.train(param, d_train, kwargs["num_round"], evallist)

    def test(self, x_test):
        return self.bst.predict(x_test)


class RandomForestSimple(Model):
    def train(self, x_train, y_train, **kwargs):
        self.clf = RandomForestClassifier(max_depth=8, random_state=0, )
        self.clf.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.clf.predict(x_test)


class KnnModel(Model):
    def train(self, x_train, y_train, **kwargs):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.model.predict(x_test)


class SimpleMlp(Model):
    def train(self, x_train, y_train, **kwargs):
        self.clf = MLPClassifier(random_state=1, max_iter=10000, hidden_layer_sizes=(200, 150, 100, 50),
                                 activation='relu',
                                 learning_rate_init=0.001)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.clf.predict(x_test)

    def get_loss(self) -> np.ndarray:
        return self.clf.loss_


class LogiticModel(Model):
    def train(self, x_train, y_train, **kwargs):
        self.logmodel = LogisticRegression(max_iter=10000)
        self.logmodel.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.logmodel.predict(x_test)


class XgboostMultiClass(Model):
    def train(self, x_train, y_train, **kwargs):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        weights = y_train.value_counts().values / y_train.size
        all_weights = np.zeros(y_train.size)
        for label in np.unique(y_train):
            all_weights[y_train == label] = weights[label]
        self.xgb = XGBClassifier(n_estimators=5000, learning_rate=0.05, gamma=0, subsample=0.7,
                                 objective="multi:softprob",
                                 num_class=len(np.unique(y_train.values)),
                                 eval_metric="mlogloss", random_state=93, use_label_encoder=False)
        self.xgb = self.xgb.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                                early_stopping_rounds=10,
                                sample_weight=all_weights)

    def predict(self, x_test) -> np.ndarray:
        return self.xgb.predict(x_test)

