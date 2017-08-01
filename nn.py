import math
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class KBSPurchaseRegressor(object):
    def __init__(self, cleaned_dataset):
        self.seed = 1
        np.random.seed(self.seed)
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.prepare_data(cleaned_dataset)
        
    def predict(self):
        return self.scaler.inverse_transform(
            self.model.predict(self.X_test)
        )

    def prepare_data(self, dataset):
        #fetch, skipping the first column (year)
        dataframe = pandas.read_csv(dataset, usecols=[1,2])

        self.X = self.scaler.fit_transform(
            dataframe["Vehicles Registered"].ix[0:].astype('float32').values.reshape(-1, 1))
        self.y = self.scaler.fit_transform(
            dataframe["Fleet Purchased"].ix[0:].astype('float32').values.reshape(-1, 1))
         
        # evaluate model with standardized dataset
        self.model = KerasRegressor(
            build_fn=self.setup_model, 
            nb_epoch=100,
            batch_size=1,
            verbose=2)

        # split into train and test sets
        # for testing model for effiency
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.33,
            random_state=self.seed)

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        self.model.fit(self.X_train, self.y_train)
  
        print "Model Train Score: %.2f" % self.model.score(self.X_train, self.y_train)
        print "Model Test Score: %.2f" % self.model.score(self.X_test, self.y_test)
       
    def setup_model(self):
        #create model
        model = Sequential([
           # create and fit the LSTM network
           LSTM(4, input_shape=(1, 1)),
           Dense(1, activation="relu"),
           Dense(1)
        ])

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

