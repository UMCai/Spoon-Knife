# Import important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import yaml


# folder to load config file
CONFIG_PATH = r'C:\Users\Shizh\OneDrive - Maastricht University\Code\Spoon-Knife\GeneralTutorials\YAML_tutorial'

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def get_argparse():
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="the config file name")
    args = vars(ap.parse_args())
    return args

def main():
    args = get_argparse()
    config_path = os.path.join(CONFIG_PATH, 'config')
    config = load_config(os.path.join(config_path, args['config']))
    # print(config)
    filename = os.path.join(CONFIG_PATH, config["data_name"])

    # load data
    data = pd.read_csv(filename)
    data.drop(columns=(config['drop_columns']),inplace=True)
    # Define X (independent variables) and y (target variable)
    X=data.drop(columns=['diagnosis'])
    y=data['diagnosis']

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=42
    )

    # call our classifer and fit to our data
    classifier = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        weights=config["weights"],
        algorithm=config["algorithm"],
        leaf_size=config["leaf_size"],
        p=config["p"],
        metric=config["metric"],
        n_jobs=config["n_jobs"],
    )
    # training the classifier
    classifier.fit(X_train, y_train)

    # test our classifier
    result = classifier.score(X_test, y_test)
    print("Accuracy score is. {:.1f}".format(result))

    # save our classifier in the model directory
    joblib.dump(classifier, os.path.join(CONFIG_PATH, config["model_name"]))


if __name__ == "__main__":
    main()