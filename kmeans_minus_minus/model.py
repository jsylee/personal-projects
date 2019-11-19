from joblib import dump
import time

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from kmeans_mm.kmeans_mm import KMeansMM

class IdentityTransformer(BaseEstimator, TransformerMixin):
    # from https://medium.com/@literallywords/sklearn-identity-transformer-fcc18bac0e98
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x


if __name__=="__main__":
    X, y = fetch_openml("mnist_784", version=1, cache=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "design_matrix",
                FeatureUnion(
                    [
                        (
                            "cluster",
                            Pipeline(
                                [
                                    ("pca", PCA()),
                                    ("k_means_mm", KMeansMM()),
                                    ("scaler", StandardScaler())
                                ]
                            ),
                        ),
                        ("identity", IdentityTransformer()),
                    ]
                ),
            ),
            ("est", RandomForestClassifier()),
        ]
    )

    params = {
        "design_matrix__cluster__pca__n_components": [5, 10, 20],
        "design_matrix__cluster__k_means_mm__n_clusters": [5, 8, 10, 20],
        "design_matrix__cluster__k_means_mm__prop_outliers": [0, 0.05],
        "est__n_estimators": [10, 50, 100, 500],
    }

    search = RandomizedSearchCV(pipeline, params, n_iter=100, iid=False, cv=5, verbose=3, n_jobs=-1)
    search.fit(X_train, y_train)
    print(search.cv_results_)
    print(search.best_estimator_)
    dump(search.best_estimator_, "best_estimator_{}.joblib".format(time.strftime("%Y%m%d_%H%M%S")))
