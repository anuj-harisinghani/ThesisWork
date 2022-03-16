from ParamsHandler import ParamsHandler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.dummy import DummyRegressor

# from keras.layers import Dense
# from keras import Sequential

import warnings
warnings.filterwarnings("ignore")


class ClassifiersFactory:
    def __init__(self, n_inputs=None, n_outputs=None):
        self.__classifiers = {
            "RandomForest": RandomForestRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "LogReg": LogisticRegression(max_iter=10000, penalty='l1', solver='liblinear'),
            "KNN": KNeighborsRegressor(),
            "SVM": LinearSVR(),
            "GradBoost": GradientBoostingRegressor(),
            "LinearReg": LinearRegression(),
            "AdaBoost": AdaBoostRegressor(),
            "Bagging": BaggingRegressor(),
            "Dummy": DummyRegressor(),

            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "Lasso_multi": MultiTaskLasso(),
            "Elastic": ElasticNet(),
            "Elastic_multi": MultiTaskElasticNet(),

            # "NeuralNetwork": ClassifiersFactory.neural_network(n_inputs, n_outputs)
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

    # @staticmethod
    # def neural_network(n_inputs, n_outputs):
    #     model = Sequential()
    #     model.add(Dense(18, input_dim=n_inputs, activation='relu'))
    #     model.add(Dense(36, input_dim=n_inputs, activation='relu'))
    #     model.add(Dense(9, input_dim=n_inputs, activation='relu'))
    #     model.add(Dense(n_outputs, activation='tanh'))
    #     model.compile(loss='mae', optimizer='adam')
    #     print(model.summary())
    #     return model
