import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC
import seaborn as sns
from contextlib import redirect_stdout
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Helpers.Constants import *
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



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


