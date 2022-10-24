from sklearn.model_selection import train_test_split

def split_data(data):


    X_train, X_val, y_train, y_val = train_test_split(data.index.values, 
                                                    data.label.values, 
                                                    test_size = 0.15, 
                                                    random_state = 2022, 
                                                    stratify = data.label.values)

    return X_train, X_val, y_train, y_val