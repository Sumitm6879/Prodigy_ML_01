# Linear Regression Model to Predict House Prices

### Building the Regression Model from scratch
----
The Basic formula for linear regression is (_Y_ = _B0_ + _B1_ * _X_) which is the slope intercept formula

![image](https://github.com/Sumitm6879/Prodigy_ML_01/assets/76704034/8b76c06b-a87d-4c0b-8cd2-448089bc8d70)

what we are trying to predict is _Y_ and what we have is _X_ 

The two constant _B0_ and _B1_ are the values we calculate based on our graphs each point

we first start at bottom at _B0_ and _B1_ Equal to 0 the we slowly increase the slop of our line using this formula

![image](https://github.com/Sumitm6879/Prodigy_ML_01/assets/76704034/f250fdcf-d8ee-4013-ba24-9733b69d4cc7)

the formula is implemented below in the function _create_model_

```py
def get_predictions(model: dict, X):
    """ 
    model: dict containg the two constants beta0 and beta1 
    x: np.array of floats containg a vector of values of independent variables

    returns an np.array of floats with values gained from the model and x
    """

    return model['beta0'] + model['beta1'] * X

def create_model(x,y):
    """
    x: np.array of floats with x intercepts 
    y: np.array of floats with y intercepts
    """

    x_bar = np.average(x)
    y_bar = np.average(y)

    top = np.sum((x-x_bar)*(y-y_bar))
    bot = np.sum((x-x_bar)**2)
    beta1 = top/bot

    beta0 = y_bar - beta1*x_bar
    model = {"beta0": beta0, "beta1": beta1}

    return model
```

### Creating Model of Multivariable Linear Regression
----
Unlike in single variable linear regression where we predict the value of _Y_ with respect to _X_, in **multivariable linear regression** we predict the value of _Y_ based on many variables like predicting the _saleprice_ based on _bathrooms_, _bedrooms_ and _area_.

The formula to calculate beta remains the same but this time we are not limited by only 2 variables we can have **N** variables 
```py
def get_prediction(model, x)
```
> This function will return a np.array of values which are calculated by the same previous formula but since we are using multivariables this function takes model as a np.array of shape (**p**, **N**) with each rows  as _B0_, _B1_, _B2_, ..., _Bn_

```py
def create_multi_model(X, Y)
```
> This function is used to generate the matrix with **p** rows and **N** columns 

```py
def get_prediction(model, x):
    """
    model: np.array of our p constants and intercept
    x: np.array of all independent values
    returns a np.array vector 
    """
    (p, n_minus_one) = x.shape

    n = n_minus_one + 1

    new_x = np.ones(shape=(p,n))
    new_x[:, 1:] = x
    return np.dot(new_x, model)

def create_multi_model(X, Y):
    """
    Creates a Model based on the given data 
    X: np.array of independent values 
    Y: np.array of observed values 
    """

    (n, p_minus_one) = X.shape

    p = p_minus_one + 1

    new_x = np.ones(shape=(n,p))
    new_x[:, 1:] = X
    return np.dot(np.dot(inv(np.dot(new_x.T, new_x)), new_x.T), Y)
```

