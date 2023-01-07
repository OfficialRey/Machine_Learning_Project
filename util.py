import math
from typing import Any, List, Union
from keras.utils import to_categorical
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_dataset(csv_file: str):
    labels, x_train, y_train = get_training_data(csv_file)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_test, y_test, labels


def encode_labels(y_train: numpy.ndarray, y_test):
    temp = numpy.append(y_train, y_test)
    encoder = LabelEncoder()
    encoder.fit(temp)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    return y_train, y_test, encoder


def one_hot_encode(y_train, y_test, classes: Union[LabelEncoder, int]):
    if isinstance(classes, LabelEncoder):
        classes = len(classes.classes_)
    y_train = to_categorical(y_train, num_classes=classes)
    y_test = to_categorical(y_test, num_classes=classes)
    return y_train, y_test


def get_training_data(csv_file: str):
    x = []
    y = []
    try:
        with open(csv_file, "r") as csv:  # Load and read CSV file
            content = csv.readlines()
            l = content[0].replace("\"", "").replace("\n", "").split(";")
            # Calculate length of x-data
            x_len = len(l) - 1
            for i in range(1, len(content)):  # Iterate over dataset and load dataset into RAM
                data = content[i].replace("\n", "").split(";")
                x.append(list(map(float, data[:x_len:])))
                y_data = data[x_len::][0]
                if y_data.isnumeric():
                    y_data = float(y_data)
                y.append(y_data)
                # x = PolynomialFeatures().fit_transform(x)
                # Different option of training data

            return l, numpy.array(x), numpy.array(y)

    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to find file {csv_file}")


def get_testing_data(x, y, test_amount: float = 0.1):
    train_len = len(x)
    test_len = int(train_len * test_amount)
    x_test = x[:test_len:]
    y_test = y[:test_len:]
    x = x[test_len:]
    y = y[test_len:]
    return x, y, x_test, y_test


def test_model(model: Any, x_test: List[List[float]], y_test: List[int]):
    x_test = list(x_test)
    y_test = list(y_test)
    predictions = model.predict(x_test)
    true_predictions = 0
    loss = 0
    dataset_len = len(x_test)
    for i in range(dataset_len):
        true_predictions += int(predictions[i] == y_test[i])
        loss += abs(predictions[i] - y_test[i]) / dataset_len
    accuracy = true_predictions / dataset_len
    mse = numpy.square(numpy.subtract(y_test, predictions)).mean()
    rmse = math.sqrt(mse)
    print(f"Tested model: {model}")
    print(f"Model accuracy: {'{:2.4f}'.format(accuracy * 100)}%")
    print(f"Average loss: {'{:2.4f}'.format(loss)}")
    print(f"Mean Squared Error: {'{:2.4f}'.format(mse)}")
    print(f"Root Mean Squared Error: {'{:2.4f}'.format(rmse)}")
    print()
