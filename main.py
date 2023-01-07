import os

from keras.metrics import RootMeanSquaredError

from models import create_models_classification, create_models_regression, create_deep_network
from util import test_model, get_dataset, one_hot_encode, encode_labels
from reinforcement_learning.env import PPOAgent

WINE_CSV = "/datasets/wine_data.csv"
IRIS_CSV = "/datasets/iris_data.csv"
BANKNOTE_CSV = "/datasets/banknote_data.csv"


def wine_data():
    # Create scikit-learn models
    x_train, y_train, x_test, y_test, labels = get_dataset(os.getcwd() + WINE_CSV)
    models = create_models_classification(x_train, y_train)
    models.extend(create_models_regression(x_train, y_train))

    # Test scikit-learn models
    for model in models:
        test_model(model, x_test, y_test)

    # Prepare data for classification learning model
    y_train, y_test, encoder = encode_labels(y_train, y_test)
    y_train, y_test = one_hot_encode(y_train, y_test, 10)

    # Create and train deep learning model
    network_classification = create_deep_network(
        x_train=x_train,
        y_train=y_train,
        input_size=11,
        output_size=10,
        size=8,
        depth=2,
        epochs=25,
        steps_per_epoch=10,
        activation="softmax",
        metrics=["accuracy", RootMeanSquaredError()],
        loss="categorical_crossentropy"
    )

    # Test deep learning model
    network_classification.test_network(x_test, y_test)

    # PPO
    ppo = PPOAgent(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_size=11,
        output_size=10
    )
    ppo.train_agent(
        epochs=50,
        steps=5,
        target_accuracy=0.8,
        timeout=5
    )
    network = ppo.get_network()
    network.test_network(x_test, y_test)


def iris_data():
    # Create scikit-learn models
    x_train, y_train, x_test, y_test, labels = get_dataset(os.getcwd() + IRIS_CSV)
    y_train, y_test, encoder = encode_labels(y_train, y_test)
    models = create_models_classification(x_train, y_train)

    # Test scikit-learn models
    for model in models:
        test_model(model, x_test, y_test)

    # Prepare data for classification learning model
    y_train, y_test = one_hot_encode(y_train, y_test, encoder)

    # Create and train deep learning model
    network_classification = create_deep_network(
        x_train=x_train,
        y_train=y_train,
        input_size=4,
        output_size=3,
        size=16,
        depth=8,
        epochs=25,
        steps_per_epoch=10,
        activation="softmax",
        metrics=["accuracy", RootMeanSquaredError()],
        loss="categorical_crossentropy"
    )

    # Test deep learning model
    network_classification.test_network(x_test, y_test)

    # PPO
    ppo = PPOAgent(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_size=4,
        output_size=3
    )
    ppo.train_agent(
        epochs=50,
        steps=5,
        target_accuracy=0.8,
        timeout=5
    )
    network = ppo.get_network()
    network.test_network(x_test, y_test)


def banknote_data():
    # Create scikit-learn models
    x_train, y_train, x_test, y_test, labels = get_dataset(os.getcwd() + BANKNOTE_CSV)
    y_train, y_test, encoder = encode_labels(y_train, y_test)
    models = create_models_classification(x_train, y_train)

    # Test scikit-learn models
    for model in models:
        test_model(model, x_test, y_test)

    # Prepare data for classification learning model
    y_train, y_test = one_hot_encode(y_train, y_test, encoder)

    # Create and train deep learning model
    network_classification = create_deep_network(
        x_train=x_train,
        y_train=y_train,
        input_size=4,
        output_size=2,
        size=16,
        depth=8,
        epochs=25,
        steps_per_epoch=10,
        activation="softmax",
        metrics=["accuracy", RootMeanSquaredError()],
        loss="categorical_crossentropy"
    )

    # Test deep learning model
    network_classification.test_network(x_test, y_test)

    # PPO
    ppo = PPOAgent(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_size=4,
        output_size=2
    )
    ppo.train_agent(
        epochs=50,
        steps=5,
        target_accuracy=0.8,
        timeout=5
    )
    network = ppo.get_network()
    network.test_network(x_test, y_test)


if __name__ == '__main__':
    wine_data()
    iris_data()
    banknote_data()
