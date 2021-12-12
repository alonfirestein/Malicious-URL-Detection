import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from Helpers.models.models import Model


class SimpleDecisionTreeClassifier(Model):
    def fit(self, x_train, y_train, **kwargs):
        self.decsion_tree_model = DecisionTreeClassifier(random_state=42,**kwargs)
        self.decsion_tree_model.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.decsion_tree_model.predict(x_test)


class XgboostModelBinary(Model):
    def fit(self, x_train, y_train, **kwargs):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        param = {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "gamma": 0, "subsample": 0.7,
                 "objective": 'binary:logistic', "scale_pos_weight": 1, "seed": 93, "eval_metric": "logloss"}
        # "colsample_bytree":0.8# evals=evals, early_stopping_rounds=10
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_eval = xgb.DMatrix(x_val, label=y_val)
        evallist = [(d_train, 'train'), (d_eval, 'eval')]
        self.bst = xgb.train(kwargs, d_train, kwargs["num_round"], evallist)

    def predict(self, x_test) -> np.ndarray:
        return self.bst.predict(x_test)


class RandomForestSimple(Model):
    def fit(self, x_train, y_train, **kwargs):
        self.clf = RandomForestClassifier(random_state=0,n_jobs=-1,**kwargs)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.clf.predict(x_test)


class KnnModel(Model):

    def fit(self, x_train, y_train, **kwargs):
        self.model = KNeighborsClassifier(**kwargs,n_jobs=-1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.model.predict(x_test)


class SimpleMlp(Model):
    def fit(self, x_train, y_train, **kwargs):
        self.clf = MLPClassifier(random_state=1, **kwargs)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.clf.predict(x_test)

    def get_loss(self) -> np.ndarray:
        return self.clf.loss_


class LogiticModel(Model):
    def fit(self, x_train, y_train, **kwargs):
        self.logmodel = LogisticRegression(**kwargs)
        self.logmodel.fit(x_train, y_train)

    def predict(self, x_test) -> np.ndarray:
        return self.logmodel.predict(x_test)


class XgboostMultiClass(Model):
    def fit(self, x_train, y_train, **kwargs):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        weights = pd.Series(y_train).value_counts().values / y_train.size
        all_weights = np.zeros(y_train.size)
        for label in np.unique(y_train):
            all_weights[y_train == label] = weights[label]
        self.xgb = XGBClassifier(n_estimators=5000, learning_rate=0.05, gamma=0, subsample=0.7,
                                 objective="multi:softprob",
                                 num_class=len(np.unique(y_train)),
                                 eval_metric="mlogloss", random_state=93, use_label_encoder=False)
        self.xgb = self.xgb.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                                early_stopping_rounds=10,
                                sample_weight=all_weights)

    def predict(self, x_test) -> np.ndarray:
        return self.xgb.predict(x_test)

