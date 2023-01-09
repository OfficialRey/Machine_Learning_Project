import math

import gym
from gym.spaces import Box, MultiDiscrete
import numpy
from keras.metrics import RootMeanSquaredError
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from models import create_deep_network


# This file is a simple Reinforcement Learning agent to discover good hyperparameters
# for a neural network

class PPOAgent:
    def __init__(self, x_train, y_train, x_test, y_test, input_size, output_size):
        self.x_test = x_test
        self.y_test = y_test
        self.is_trained = False
        self.env = GymEnv(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_size=input_size,
            output_size=output_size
        )

    def train_agent(self, training_interval=25, epochs=20, steps=5, target_accuracy=0.8, timeout=2):
        model = PPO(MlpPolicy, self.env, n_epochs=epochs, n_steps=steps, verbose=1)
        for _ in range(timeout):
            model.learn(total_timesteps=training_interval)
            network = self.env.get_model()
            _, accuracy, _ = network.test_network(
                self.x_test,
                self.y_test,
                0
            )
            if accuracy > target_accuracy:
                break
        print("Best found hyper parameters: ")
        self.env.render(force=True)
        self.is_trained = True

    def get_network(self, target_accuracy=0.8, timeout=10):
        if self.is_trained:
            network = None
            for _ in range(timeout):
                network = self.env.get_model()
                loss, accuracy, rmse = network.test_network(
                    self.x_test,
                    self.y_test,
                    0
                )
                if accuracy >= target_accuracy:
                    return network
            return network
        else:
            raise RuntimeError("Trying to access untrained network.")


# Here we create an OpenAI Gym environment to be used for the PPOAgent to train on
class GymEnv(gym.Env):
    loss: float
    rmse: float
    accuracy: float
    size: int
    depth: int
    epochs: int
    steps_per_epoch: int

    last_best_accuracy: float
    best_accuracy: float
    best_size: int
    best_depth: int
    best_epochs: int
    best_steps_per_epoch: int

    # Here we set some parameters to be used for training
    def __init__(self, x_train, y_train, x_test, y_test, input_size, output_size):
        # Training data for trained model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_size = input_size
        self.output_size = output_size

        # Update info
        self.last_best_accuracy = 0
        self.best_accuracy = 0
        self.best_size = 1
        self.best_depth = 1
        self.best_epochs = 1
        self.best_steps_per_epoch = 1

        # Init default values
        self.loss = 0
        self.accuracy = 0
        self.rmse = 0
        self.size = 1
        self.depth = 1
        self.epochs = 1
        self.steps_per_epoch = 1

        # Set the observation space to the hyperparameters of the neural network and its metrics:
        # --> Loss, Accuracy, Rmse, Size, Depth, Epochs, StepsPerEpoch
        self.observation_space = Box(low=numpy.array([0, 0, 0, 1, 1, 1, 1]),
                                     # We want an upper limit for the size depth, epochs and steps per epoch
                                     # A depth of 20 might still be a bit overkill, but again -
                                     # these are hyperparameters
                                     high=numpy.array([math.inf, 1, math.inf, 20, 20, 50, 25]))
        self.action_space = MultiDiscrete([32, 8, len(x_test) // 5 + 1, len(x_test) // 5 + 1], dtype=int)

    # Take a step in the environment
    # We create the network according to the networks actions, calculate its performance and reward the bot
    # according to its performance
    # As an observation we hand the bot its own actions as well as the metrics of the network
    def step(self, action):
        self.size, self.depth, self.epochs, self.steps_per_epoch = action
        self.size += 1
        self.depth += 1
        self.epochs += 1
        self.steps_per_epoch += 1
        model = create_deep_network(
            x_train=self.x_train,
            y_train=self.y_train,
            input_size=self.input_size,
            output_size=self.output_size,
            size=self.size,
            depth=self.depth,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            activation="softmax",
            metrics=["accuracy", RootMeanSquaredError()],
            loss="categorical_crossentropy",
            verbose=0)

        self.loss, self.accuracy, self.rmse = model.test_network(self.x_test, self.y_test, verbose=0)
        reward = self.accuracy - self.loss - self.rmse
        if self.accuracy > self.best_accuracy:
            self.best_accuracy = self.accuracy
        self.render()

        # Output actions
        print("\r", end="")
        print(f"Current actions: {action} | Accuracy: {self.accuracy}", end="")

        return self.get_obs(), reward, False, {}

    def reset(self):
        return self.get_obs()

    def get_obs(self):
        return [self.loss, self.accuracy, self.rmse, self.size, self.depth, self.epochs, self.steps_per_epoch]

    # Show some basic output in the console if wished
    def render(self, mode="human", force=False):
        update = force
        if self.last_best_accuracy != self.best_accuracy:
            self.best_size = self.size
            self.best_depth = self.depth
            self.best_epochs = self.epochs
            self.best_steps_per_epoch = self.steps_per_epoch
            update = True
        self.last_best_accuracy = self.best_accuracy
        if update:
            print()
            print(f"Best Results:")
            print(f"Accuracy: {self.best_accuracy}")
            print(f"Size: {self.best_size}")
            print(f"Depth: {self.best_depth}")
            print(f"Epochs: {self.best_epochs}")
            print(f"Steps: {self.best_steps_per_epoch}")
            print()

    # Create a model based on what the ppo agent finds the best hyperparameters
    def get_model(self):
        return create_deep_network(
            x_train=self.x_train,
            y_train=self.y_train,
            input_size=self.input_size,
            output_size=self.output_size,
            size=self.best_size,
            depth=self.best_depth,
            epochs=self.best_epochs,
            steps_per_epoch=self.best_steps_per_epoch,
            activation="softmax",
            metrics=["accuracy", RootMeanSquaredError()],
            loss="categorical_crossentropy",
            verbose=0)
