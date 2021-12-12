from Helpers.Constants import *
from Helpers.models import *
from Helpers.preprocess import *


def experiments():

    x_train, x_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.1, random_state=42)
    XgboostModelBinary().pipeline(x_train, y_train, x_test, y_test, model_name="XgboostModelBinary",
                                  do_cross_validation=True)
    SimpleDecisionTreeClassifier().pipeline(x_train, y_train, x_test, y_test,
                                            model_name="SimpleDecisionTreeClassifier", do_cross_validation=True)
    SimpleMlp().pipeline(x_train, y_train, x_test, y_test, "mlp")
