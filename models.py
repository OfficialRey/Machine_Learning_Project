from typing import List, Union, Any

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from deep_learning import DeepNetwork


# This file is used to create different machine learning models

# The next few functions are self explanatory and therefore not described closer

def create_svm(x_train, y_train):
    svm = SVC(gamma='auto')
    svm.fit(x_train, y_train)
    return svm


def create_naive_bayes(x_train, y_train):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    return gnb


def create_k_nearest_neighbour(x_train, y_train, neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(x_train, y_train)
    return knn


def create_decision_tree(x_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    return dt


def create_random_forest(x_train, y_train, depth=8):
    rf = RandomForestClassifier(max_depth=depth, random_state=0)
    rf.fit(x_train, y_train)
    return rf


def create_logistic_regression(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    return lr


def create_deep_network(x_train, y_train, input_size, output_size, size: int, depth: int, epochs: int,
                        steps_per_epoch: int, activation: Union[Any, str],
                        metrics: Union[List[str], List[Any]], loss: Union[Any, str], verbose="auto"):
    network = DeepNetwork(x_train, y_train, input_size, output_size, size, depth, activation, metrics, loss, epochs,
                          steps_per_epoch, verbose)
    return network


# Creates and groups classification models to be used in a for loop
def create_models_classification(x_train, y_train):
    # Trains classification models

    svm = create_svm(x_train, y_train)
    gnb = create_naive_bayes(x_train, y_train)
    knn = create_k_nearest_neighbour(x_train, y_train, neighbors=5)
    dt = create_decision_tree(x_train, y_train)
    rf = create_random_forest(x_train, y_train)

    return [svm, gnb, knn, dt, rf]


# Unused
def create_models_regression(x_train, y_train):
    # Trains regression models

    lo = create_logistic_regression(x_train, y_train)
    return [lo]
