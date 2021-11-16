from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR


import warnings
warnings.filterwarnings("ignore")


class ClassifiersFactory:
    def __init__(self):
        self.__classifiers = {
            "RandomForest": RandomForestRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "LogReg": LogisticRegression(max_iter=100000),
            "KNN": KNeighborsRegressor(),
            "SVM": LinearSVR(),
            "GradBoost": GradientBoostingRegressor()
        }

    def get_models(self, classifiers: list) -> list:
        models = []
        for classifier in classifiers:
            if classifier not in self.__classifiers.keys():
                raise ValueError("Invalid classifier '{}'! Supported classifiers are: {}"
                                 .format(classifier, self.__classifiers.keys()))
            models.append(self.__classifiers[classifier])
        return models

    def get_model(self, classifier: str) -> object:
        if classifier not in self.__classifiers.keys():
            raise ValueError("Invalid classifier '{}'! Supported classifiers are: {}"
                             .format(classifier, self.__classifiers.keys()))
        return self.__classifiers[classifier]
