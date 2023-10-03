# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 
 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: KOWSALYA M
RegisterNumber: 212222230069
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m) * np.sum(square_err)

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha * 1/m * error
    theta-=descent
    j_history.append(computeCost(X,y,theta))
  return theta,j_history

theta,j_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70000,we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![linear regression using gradient descent](sam.png)
![4 1](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/fc874ffb-6422-42a9-ae82-669630c1215a)
### profit Prediction graph:
![4 2](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/d05d822a-8ffe-43ce-941d-e96aac549feb)
### Compute Cost Value:
![4 3](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/0f6f9002-9a53-43ef-afc4-7602449dd8dc)
### h(x) Value and Cost function using Gradient graph:
![4 4](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/9e6e6663-47ed-40e7-8786-04daae92909e)
### Profit prediction graph:
![4 5](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/56f3a476-7b0c-4b9c-a887-338b3191b406)
### 
![4 6](https://github.com/Kowsalyasathya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118671457/7e757ea6-6a4b-458c-b12b-e68852823db8)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
