from typing import Any, List, Union
from keras import Model, Input, Sequential
from keras.layers import Dense, ReLU, Activation
from keras.optimizers import Adam


class DeepNetwork:
    model: Sequential

    def __init__(self, x_train, y_train, input_size, output_size, size: int, depth: int,
                 output_activation_function: str,
                 metrics: List[str], loss: Union[str, Any], epochs: int, steps_per_epoch: int, verbose="auto"):
        if metrics is None:
            metrics = []
        self._create_dense_network(
            input_size=input_size,
            output_size=output_size,
            size=size,
            depth=depth,
            output_activation_function=output_activation_function,
            metrics=metrics,
            loss=loss
        )
        self.train_network(x_train, y_train, epochs, steps_per_epoch, verbose=verbose)

    def _create_dense_network(self, input_size: Any, output_size: Any, size: int, depth: int,
                              output_activation_function: str, loss: str, metrics: List[str]):
        input_layer = Input(input_size)
        x = input_layer
        for _ in range(depth):
            x = Dense(size)(x)
            x = ReLU()(x)
        x = Dense(output_size)(x)
        x = Activation(output_activation_function)(x)
        self.model = Model(inputs=input_layer, outputs=x)

        self.model.compile(
            optimizer=Adam(
                learning_rate=0.001,
                epsilon=1e-07
            ),
            loss=loss,
            metrics=metrics
        )

    def train_network(self, x_train, y_train, epochs: int, steps_per_epoch: int, verbose="auto"):
        self.model.fit(x_train, y_train, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose)

    def test_network(self, x_test, y_test, verbose=1):
        if verbose > 0:
            print("Testing model.")
        loss, accuracy, rmse = self.model.evaluate(x_test, y_test, steps=len(y_test), verbose=verbose)
        return loss, accuracy, rmse
