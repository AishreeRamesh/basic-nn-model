# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction. 

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as `relu` and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
``` python3
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("data.csv")

df.head()

x=df[["INPUT"]].values

y=df[["OUTPUT"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

import tensorflow as tf

model=tf.keras.Sequential([tf.keras.layers.Dense(8,activation='relu'),
                           tf.keras.layers.Dense(16,activation='relu'),
                           tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="adam",metrics=["mse"])

history=model.fit(x_train,y_train,epochs=1000)

import numpy as np

x_test

preds=model.predict(x_test)
np.round(preds)

tf.round(model.predict([[20]]))

pd.DataFrame(history.history).plot()

r=tf.keras.metrics.RootMeanSquaredError()
r(y_test,preds)

```
## Dataset Information

![image](https://user-images.githubusercontent.com/70213227/187081607-1ba5ddb6-0af1-48e5-a1d8-ef461012f5e5.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/70213227/187081622-5e09d768-dd62-4c54-a57e-fdc990c8966f.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/70213227/187081639-f1b517b9-bcfa-43ad-9e53-c12ae4f84576.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/70213227/187081690-9feb6f22-98d6-4832-b32e-cec9a43fc3c4.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
