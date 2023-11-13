#mport important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
sys_path = r'C:\Users\Shizh\OneDrive - Maastricht University\Code\Spoon-Knife\GeneralTutorials\YAML_tutorial'

#path to the dataset
filename = os.path.join(sys_path, 'data.csv')

#load data 
data = pd.read_csv(filename)

data.drop(columns=(['id','Unnamed: 32']),inplace=True)
# Define X (independent variables) and y (target variable)
X=data.drop(columns=['diagnosis'])
y=data['diagnosis']
#split data into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# call our classifer and fit to our data
classifier = KNeighborsClassifier(n_neighbors=5, weights="uniform",
								 algorithm = "auto", leaf_size = 25,
								 p=1, metric="minkowski", n_jobs=-1)
								 
#training the classifier
classifier.fit(X_train, y_train)

#test our classifier 
result = classifier.score(X_test, y_test)
print("Accuracy score is. {:.1f}".format(result))

#save our classifier in the model directory
model_name = "KNN_classifier"
joblib.dump(classifier, os.path.join(sys_path, '{}.pkl'.format(model_name)))