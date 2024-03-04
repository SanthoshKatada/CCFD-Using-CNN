import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Importing the csv File 
data = pd.read_csv("creditcard.csv")

# Balencing the Dataset
non_fraud = data[data['Class']==0]
fraud = data[data['Class']==1]

# Randomly picking 492 values to balence the dataset.
non_fraud = non_fraud.sample(fraud.shape[0])

# Merging the fraud and non-fraud datasets to get a balenced dataset
data = fraud._append(non_fraud, ignore_index=True)

# Feature Space Definition
# Rows and Columns are called Features
# Feature Space is all the rows & columns needed to train the model excluding the target rows & coloumns (Here target coloums is Class)

# X has everything except class
x = data.drop('Class' , axis=1)

# Y has nothing but class
y = data['Class']

# Train_test_split is used to split Datasets into  trainig and testing datasets
# x - train(80%) & test(20%)
# y - train(80%) & test(20%)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0, stratify=y)

# As the varience is very high in the dataset we have to standardize the dataset
# varience - A measurment of how far each number is to the average of the dataset , Basically it denotes the spread of the values in Dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Converting the data into numpy array as CNN takes a 3D array as input
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#Reshaping the x dataset to 3Dimensions
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# Building the CNN Model
epochs = 50
model = Sequential()

#First CNN layer
model.add(Conv1D(filters=32 , kernel_size=2 , activation='relu', input_shape= x_train[0].shape))

model.add(BatchNormalization())

model.add(MaxPool1D(pool_size=2))

#Dropout Layer
model.add(Dropout(0.2))

#Second CNN Layer
model.add(Conv1D(filters=64 , kernel_size=2 , activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool1D(pool_size=2))

#Dropout Layer
model.add(Dropout(0.3))

#Flatten Layer
model.add(Flatten())

#1st Dense Layer
model.add(Dense(units=64 , activation='relu'))

model.add(Dropout(0.3))

#2nd Dense Layer (or) Output Layer
model.add(Dense(units=1, activation='sigmoid'))

#Compiling the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(x_train , y_train , epochs=epochs , validation_data=(x_test , y_test) , verbose=1)

# Web app

st.title("Credit Card Fraud Detection Model Using CNN")
input_df = st.text_input("Enter all the Required Values")
input_df_splitted = input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splitted , dtype=np.float32)
    prediction = model.predict(features.reshape(1,-1))

    if prediction > 0.5:
        st.write("Fraud transaction")
    else:
        st.write("Non Fraud Transaction")

