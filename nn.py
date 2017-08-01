import math
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class KBSPurchaseRegressor(object):
    def __init__(self, cleaned_dataset):
        self.seed = 100
        np.random.seed(self.seed)

        self.prepare_data(cleaned_dataset)
        
    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict(np.array(X))

    def prepare_data(self, dataset):
        dataframe = pandas.read_csv(dataset)
        forecast_column = "Fleet Purchased"
 
        self.X = np.array(dataframe["Vehicles Registered"])
        self.y = np.array(dataframe[forecast_column])

        # evaluate model with standardized dataset
        self.model = KerasRegressor(
            build_fn=self.setup_model, 
            nb_epoch=100, 
            verbose=0)

        # test model for effiency
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.33, 
            random_state=self.seed)

        # scale the items to avoid exaggerated figures
        self.X_train = scale(self.X_train)
        self.model.fit(self.X_train, self.y_train)
        
        print "Model Score: %.2f" % self.model.score(self.X_test, self.y_test)

    def setup_model(self):
        #create model
        model = Sequential([
           Dense(1, input_dim=1, kernel_initializer='normal', activation='relu'),
           Dense(1, kernel_initializer='normal')
        ])

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

