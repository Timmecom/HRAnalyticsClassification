#import all the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#import the dataset
X_train= pd.read_csv(r"X_train.csv")
y_train= pd.read_csv(r"y_train.csv")

reg = RandomForestClassifier(random_state=0)
reg.fit(X_train, np.ravel(y_train))
