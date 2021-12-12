from tqdm import tqdm

from Helpers.Constants import *
from Helpers.preprocess.main import *
from Helpers.models.simple_ml import *
from Helpers.models.models import Model
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


def experiments(model: Model, x_train, x_test, y_train, y_test, model_name: str, do_cross_validation, **kwargs):
    model.pipeline(x_train, y_train, x_test, y_test, model_name, do_cross_validation=do_cross_validation, **kwargs)
    pca = PCA(n_components=0.99)
    pca.fit(x_train)
    model.pipeline(pca.transform(x_train), y_train, pca.transform(x_test), y_test,
                   f"{model_name}-pca-{len(pca.components_)}", do_cross_validation=do_cross_validation, **kwargs)


def main():
    df = preprocess(data_path="../Data/malicious_phish.csv", save_to_path="cleaned_debug.csv", cache=True,
                    features_extractors=(get_lexical_url_features,))
    X_train, X_test, y_train, y_test = scale_data(df)
    # df.sample(50000).to_csv("cleaned_debug.csv")
    models = [
        (XgboostMultiClass(), "Xgboost", {}),
        (LogiticModel(), "LogisticRegression", {"max_iter": 10000}),
        (KnnModel(), "KNN", {"n_neighbors": 15}),
        (RandomForestSimple(), "RandomForest", {"max_depth": 13}),
        (SimpleMlp(), "MLP", {"max_iter": 10000, "hidden_layer_sizes": (200, 150, 100, 50),
                              "activation": 'relu',
                              "learning_rate_init": 0.001}), ]
    # voter = VotingClassifier(estimators=[(a[1], a[0]) for a in models], voting='soft')

    for model, model_name, params in tqdm(models):
        experiments(model, X_train, X_test, y_train, y_test, model_name, do_cross_validation=False, **params)


if __name__ == '__main__':
    main()
