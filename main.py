from nn import KBSPurchaseRegressor

def main():
    # I decided to clean the data based of actual registration
    # data. I took time to study the trends.
    # I also inferred when the fleet might have been bought.
    training_data = "fleet2.csv"

    regressor = KBSPurchaseRegressor(training_data)
    purchase_prediction = regressor.predict()
    
    print "The number of likely purchase over the next year is: ", purchase_prediction

if __name__ == "__main__":
    main()
