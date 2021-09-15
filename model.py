#import all the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

#import the dataset
trainset= pd.read_csv(r"trainset_cleaned.csv")
#replace the values in the "region" column with only numbers
trainset["region"] = trainset["region"].str.split("_", expand=True)[1]
#change the data type of "region" from "object" to "int32"
trainset["region"] = trainset["region"].astype("int32")
def department_int(dept):
    dept_dict = {"Sales & Marketing": 1, "Operations":2, "Procurement":3, "Technology":4, "Analytics":5, "Finance":6, "HR":7, "Legal":8, "R&D":9}
    return dept_dict[dept]
def education_int(edu):
    edu_dict = {"Master's & above": 1, "Bachelor's":2, "Below Secondary":3}
    return edu_dict[edu]
def gender_int(gender):
    gender_dict = {"m": 1, "f":2}
    return gender_dict[gender]
def channel_int(channel):
    channel_dict = {"sourcing": 1, "referred":2, "other":3}
    return channel_dict[channel]
trainset["department"] = trainset["department"].apply(lambda x: department_int(x))
trainset["education"] = trainset["education"].apply(lambda x: education_int(x))
trainset["gender"] = trainset["gender"].apply(lambda x: gender_int(x))
trainset["recruitment_channel"] = trainset["recruitment_channel"].apply(lambda x: channel_int(x))
#Feature Selection based on the result of the heatmap including the categorical variables
X = trainset.drop(["employee_id", "no_of_trainings", "length_of_service", "is_promoted"], axis=1)
y = trainset["is_promoted"]
#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=1200)
# Undersample and plot imbalanced dataset with the neighborhood cleaning rule
# oversample
over = RandomOverSampler(sampling_strategy=0.1)
X_train, y_train = over.fit_resample(X_train, y_train)
# undersample
under = RandomUnderSampler(sampling_strategy=0.5)
X_train, y_train = under.fit_resample(X_train, y_train)
reg = RandomForestClassifier(random_state=0)
reg.fit(X_train, y_train)