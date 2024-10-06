# CS 4375.001
# Neural Network on Wine Quality Dataset
# Saidarsh Tukkadi / SXT200072
# Caleb Kim / CJK200004

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import logging

# suppress tensorflow logs to avoid unnecessary output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# suppress warnings of type 'UserWarning' from tensorflow/keras
warnings.filterwarnings('ignore', category=UserWarning)

# suppress logging from tensorflow to avoid cluttered console outputs
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class NeuralNet:
    def __init__(self, data_url, header=True):
        # load the dataset from the provided url using pandas
        self.raw_input = pd.read_csv(data_url, delimiter=';')
        # print the column names loaded from the dataset
        print("Columns loaded:", self.raw_input.columns)

    def preprocess(self):
        # remove any rows with missing values in the dataset
        self.raw_input.dropna(inplace=True)
        
        # separate features and target variable ('quality') from the dataset
        if 'quality' in self.raw_input.columns:
            X = self.raw_input.drop('quality', axis=1)
            y = self.raw_input['quality']
        else:
            # raise an error if 'quality' column is not found
            raise KeyError("Column 'quality' not found in the dataset.")
        
        # scale the feature values using standard scaling
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.y = y
        
        # split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42)

    def build_model(self, activation='relu', learning_rate=0.01, hidden_layers=2):
        # create a sequential neural network model
        model = Sequential()
        # add the first layer with 64 neurons and specified activation function
        model.add(Dense(64, activation=activation, input_shape=(self.X_train.shape[1],)))
        
        # add hidden layers based on the provided number of hidden layers
        for _ in range(hidden_layers):
            model.add(Dense(32, activation=activation))
        
        # add the output layer with 1 neuron (for regression)
        model.add(Dense(1))
        
        # configure the optimizer and compile the model with mse loss function
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return model

    def train_evaluate(self):
        # define various hyperparameters for training the model
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        epochs_list = [100, 200]
        hidden_layers_options = [2, 3]
        
        # dictionary to store history of training for plotting
        history_data = {}
        # list to store final results (train loss, val loss, r-squared)
        results = []
        
        # configure early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # iterate over all combinations of activations, learning rates, epochs, and hidden layers
        for activation in activations:
            for lr in learning_rates:
                for epochs in epochs_list:
                    for hidden_layers in hidden_layers_options:
                        # build the model with the specified hyperparameters
                        model = self.build_model(activation=activation, learning_rate=lr, hidden_layers=hidden_layers)
                        
                        # train the model and store training history
                        history = model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test), 
                                            batch_size=32, verbose=0, callbacks=[early_stopping])
                        
                        # key to identify the configuration
                        key = f"activation={activation}, lr={lr}, epochs={epochs}, hidden_layers={hidden_layers}"
                        history_data[key] = history
                        
                        # get the final training and validation losses
                        train_loss = history.history['loss'][-1]
                        val_loss = history.history['val_loss'][-1]
                        
                        # generate predictions and calculate r-squared value
                        y_pred = model.predict(self.X_test)
                        r2 = r2_score(self.y_test, y_pred)
                        
                        # append the results to the list
                        results.append({
                            'Activation': activation,
                            'Learning Rate': lr,
                            'Epochs': epochs,
                            'Hidden Layers': hidden_layers,
                            'Train Loss': train_loss,
                            'Validation Loss': val_loss,
                            'R-squared': r2
                        })
                        
                        # print the results for each configuration
                        print(f"{key} -> Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, R-squared: {r2:.4f}")

        # convert the results list into a pandas dataframe and print the summary
        results_df = pd.DataFrame(results)
        print("\nFinal Results Summary:")
        print(results_df)
        
        # plot the training history
        self.plot_model_history(history_data)

    def plot_model_history(self, history_data):
        # iterate over each activation function and plot the validation loss
        for activation in ['sigmoid', 'tanh', 'relu']:
            plt.figure(figsize=(12, 8))
            for key, history in history_data.items():
                if f"activation={activation}" in key:
                    plt.plot(history.history['val_loss'], label=key)
            plt.title(f'Validation Loss for {activation} Activation Function')
            plt.ylabel('Validation Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            plt.show()

if __name__ == "__main__":
    # url of the dataset used for training the neural network
    url = "https://raw.githubusercontent.com/saidarsht/Wine-Quality-Dataset/main/winequality-red.csv"
    
    # create an instance of the neural network class and preprocess the dataset
    neural_network = NeuralNet(url)
    neural_network.preprocess()
    
    # train the model and evaluate its performance
    neural_network.train_evaluate()
