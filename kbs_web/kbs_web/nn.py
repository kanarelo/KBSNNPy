import re
import os
import csv
import math
import string
import itertools
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
References:
http://deeplearning4j.org
https://keras.io/getting-started/sequential-model-guide/
https://keras.io/optimizers/
https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
https://keras.io/getting-started/sequential-model-guide/
https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py

KNBS:
https://www.knbs.or.ke/download/lei-december-2015/
https://www.knbs.or.ke/download/lei-december-2014/
https://www.knbs.or.ke/download/lei-december-2013/
https://www.knbs.or.ke/download/lei-december-2012/
https://www.knbs.or.ke/download/lei-december-2011/
https://www.knbs.or.ke/download/lei-december-2010/
https://www.knbs.or.ke/download/lei-december-2009/
https://www.knbs.or.ke/download/lei-december-2008/
https://www.knbs.or.ke/download/lei-december-2007/
"""

class KBSPurchaseRegressor(object):
    def __init__(self, fleet_data=None, retrain=False):
        self.seed = 1
        np.random.seed(self.seed)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if fleet_data is None:
           fleet_data = "kbs_web/fleet_with_years.csv"
        self.fleet_data = fleet_data

    def predict(self):
        return self.scaler.inverse_transform(
            self.model.predict(self.X_test))

    def prepare_data(self):
        """
        Read fleet_with_years.csv and train model with data
        """
        cleaned_data = list(self.get_cleaned_data())

        vehicles_registered = np.array(list((v,) for (y, v, f) in cleaned_data))
        fleet_purchased = np.array(list((f,) for (y, v, f) in cleaned_data))

        #get items of column "Vehicles Registered", and reshape to 2D
        self.X = self.scaler.fit_transform(
            np.array(vehicles_registered).astype('float32').reshape(-1, 1))
        #get items of column "Fleet Purchased", and reshape to 2D
        self.y = self.scaler.fit_transform(
            np.array(fleet_purchased).astype('float32').reshape(-1, 1))

    def train_model(self):
        """
        Create a keras model and train it using data provided
        """
        self.prepare_data()

        # evaluate model with standardized dataset
        self.model = KerasRegressor(
            build_fn=self.setup_model,
            nb_epoch=200,
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

        print "Model Test Score: %.2f" % self.model.score(self.X_test, self.y_test)

    def setup_model(self):
        """
        Create a Keras sequential neural network model
        Has an input LSTM recurrent layer, with 4 units/nodes
        And and output dense layer 
        """
        #create model
        model = Sequential([
           # create and fit the LSTM network
           LSTM(4, input_shape=(1, 1)),
           Dense(1, activation="relu")
        ])

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def get_sales_by_year(self):
        """
        observe dataset of fleet and calculate the possible
        sale of cars by year

        @returns a dictionary of sales by year;
           keys: string of year
           values: tuple pair of sales and frequency
        """
        rows = sorted(list((" ".join(r[:2]), r[2]) for r in csv.reader(open(self.fleet_data)))[1:])
        dataset = [(
               rows[i][0], 
               int(rows[i][1]), 
               self.calculate_number_of_plates(rows[i][0], rows[i + 1][0])
            ) for i in range(len(rows) - 1)]

        sales_by_year = {}
        for plate, year, number_of_cars in dataset:
            previous_values = sales_by_year.get(year)

            if previous_values is not None:
                next_values = (previous_values[0] + number_of_cars, previous_values[1] + 1)
            else:
                next_values = (number_of_cars, 1)

            sales_by_year[year] = next_values

        return sales_by_year

    def get_cleaned_data(self):
        """
        Calculates frequency of purchases
        and returns a iterable of
        (year, total purchase, frequency)

        @returns a generator of tuples
        """
        data =  self.get_sales_by_year()
        #sort by year
        items = sorted(data.items(), key=lambda x:x[0])

        for year, (total_purchases, frequency) in items:
            yield year, total_purchases, frequency

    def extract_number_plate(self, sentence):
        """
        Take in a string and extract a
        Kenyan vehicle number plate.

        >>> extract_number_plate("KBL 468B")
        ["KBL 468B"]
        >>> extract_number_plate("GBS 333")
        []
        >>> extract_number_plate("KRE 635")
        ["KRE 635"]
         >>> extract_number_plate("KTB 222")

        @returns a list of number plates
        """

        number_plate_regex_pattern = r"(K[A-Z]{2}\ [0-9]{3}[A-Z]{0,1})"
        return re.findall(number_plate_regex_pattern, sentence.upper())

    _cache = None
    def calculate_number_of_plates(self, plate_a, plate_b):
        """
        Calculate the number of vehicle purchases between two number plates

        Example:
        >>>  calculate_number_of_plates("KBS 200K", "KCL 444J")
        329900

        @returns an integer; number of vehicles
        """

        def generate_plate_numbers():
            """
            Simulates the plate number generator, returning
            an iterable with all the possible plate numbers 
            in Kenya.

            @returns a generator; list(generate_plate_numbers()) to get a list
            """
            for a in string.ascii_uppercase:
                for b in string.ascii_uppercase:
                    for i in range(10):
                        for j in range(10):
                            for k in range(10):
                                for c in string.ascii_uppercase:
                                    yield "K%s%s %d%d%d%s" % (a, b, i, j, k, c)

        if KBSPurchaseRegressor._cache is None:
            KBSPurchaseRegressor._cache = list(generate_plate_numbers())
        plates = KBSPurchaseRegressor._cache
        
        return abs(plates.index(plate_a) - plates.index(plate_b))

if __name__ == "__main__":
    training_data = "kbs_web/fleet2.csv"
    
    regressor = KBSPurchaseRegressor(training_data)
    purchase_prediction = regressor.predict()

    print "The number of likely purchase over the next 3 years is: ", map(int, purchase_prediction)
