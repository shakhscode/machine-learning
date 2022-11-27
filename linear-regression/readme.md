Note: The markdown cells of the original file is rendered either in Jupyter lab or VSCode. In my preview it is working well. But github markdown preview doesn't support all latex or markdown facilities. To get the original preview please open the file [BasicTheory.ipynb](BasicTheory.ipynb)

# Regression 
A predictive model to establish the relationship between a dependent variable and independent variable(s) to predict continous output for a real given value as input.

# Linear Regression 
When the dependent variable and the independent variable(s) are linearely related then it is called linear regression. 
In linear regression data is modeled using a linear equation. For example in a dataset with **m** no. of features the output for a particular instance is given by 
$$ y = w_0+w_1x_1+w_2x_2 +...+ w_mx_m$$
Where $ w_0,w_1,...,w_m $ are randomly introduced coefficients. In statistics these coefficients are calcualted using certain formulae but in Machine learning these are first introduced randomly and then values are updated using an optimization algorithm to get optimal coefficients for the best fit.

## Concept of linear regression.
In linear regression, we assume that output variable $ y $ is dependent on feature variables $ X $ in a linear manner. The assumption is called linear hypothesis and for a particular data point it is expressed as  $$ h_i(w,x) = w_0x_{0,i}+w_1x_{1,i}+w_2x_{2,i}+.....+ w_mx_{m,i} $$ 
$$ or, h_i(w,x) = \sum_{j=0}^{m} w_jx_{j,i} \ \ \ ,\ Here  \ x_{j,i} is \ the \ i^{th} \ instance \ of \ j^{th} \ feature.$$
for total $ m $ no. of feature.The feature $ x_{0,i} $ is an added bias and $ x_{0,i} = 1 $ for any $ i = 1,2,3,....,n $. Here $ n $ is the total no. of data points in the dataset. 

### Error and Cost function:
Our actual output is $ y_i $ and output accroding to hypothesis is $ h_i $. So error in hypothesis is 
$$ error = (y_i - h_i) $$ 
And for all instances total error is given by mean square error (MSE)

$$  MSE = \sum_{i=1}^{n}(y_i-h_i)^2 $$

In linear regression optimization goal is to find best coefficients of regression so that this error is minimum. Best values are obtained in an iterative process and this error function determines the computional cost of the overall process. Thus error function is also called cost function which is expressed as 
$$ J(w) = \frac{1}{2n}\sum_{i=1}^{n}(y_i-h_1)^2 $$
And optimization or learning objective is 
$$ minimize \ J(w) = minimize \ \frac{1}{2n}\sum_{i=1}^{n}(y_i-h_1)^2 $$

### Gradient descent 
> In simple words, a gradient is a measure of how much the output of a function changes if we change the inputs. (By Lex Fridman, MIT)

> In mathematical term, gradient descent is an optimization algorithm that numerically finds the local minimum of a multivariable function.

In machine learning an iterative gradient descent algorithm is used to minimize the cost function.
The partial derivatives 

$$\frac{\partial J(w)}{\partial w_0},\frac{\partial J(w)}{\partial w_1},...\frac{\partial J(w)}{\partial w_m} $$ 
are considered as gradients and updated in each iteration using the algorithm
$$Repeat \ until \ convergence
\left\{\begin{matrix}
w_j := \alpha_j - lr \times \frac{\partial J(w)}{\partial w_j}
\end{matrix}\right.
\\
for \ j=0,1,2,3,...,m
 $$
Here $ lr = learning \ rate $. For example coeffiecients $ w_0,w_1 $ will be updated in each iteration as 
$$ temp0 = w_0 - lr \times \frac{\partial J(w)}{\partial w_0} \\
temp1= w_1 - lr \times \frac{\partial J(w)}{\partial w_1} \\
w_0 = temp0 \\
w_1 = temp1
$$


## Linear regression in Vectorized form

If there are total **m no. of features and n no. of instances** then the feature matrix $ X $ is given by 

$$X= \begin{bmatrix}
x_{0,1}=1 & x_{1,1} & x_{2,1} & ... & ... & x_{m,1}\\ 
x_{0,2}=1 & x_{1,2} & x_{2,2} & ... & ... & x_{m,2} \\ 
x_{0,3}=1 & x_{1,3} & x_{2,3} & ... & ... & x_{m,3}\\ 
 ... & ...& ... & ... & ... & ...\\
... & ...& ... & ... & ... & ...\\
x_{0,n}=1 & x_{1,n} & x_{2,n} & ... & ... & x_{m,n}
\end{bmatrix}_{n\times(m+1)}
$$

$$ x_{j,i} = j^{th} \ feature, \ i^{th} \ instance$$
In the feature matrix a cloumn represents a feature and a row represents a data point.

The coefficient matrix is introduced as 
$$W= \begin{bmatrix}
w_0\\ w_1
\\ w_2
\\ ...
\\ ...
\\ 
w_m
\end{bmatrix}_{(m+1)\times 1}$$

So hypothesis is given by 
$$ h(w) = XW\\
h(w)=\begin{bmatrix}
x_{0,1}=1 & x_{1,1} & x_{2,1} & ... & ... & x_{m,1}\\ 
x_{0,2}=1 & x_{1,2} & x_{2,2} & ... & ... & x_{m,2} \\ 
x_{0,3}=1 & x_{1,3} & x_{2,3} & ... & ... & x_{m,3}\\ 
 ... & ...& ... & ... & ... & ...\\
... & ...& ... & ... & ... & ...\\
x_{0,n}=1 & x_{1,n} & x_{2,n} & ... & ... & x_{m,n}
\end{bmatrix}_{n\times(m+1)} \times 
\begin{bmatrix}
w_0\\ w_1
\\ w_2
\\ ...
\\ ...
\\ 
w_m
\end{bmatrix}_{(m+1)\times 1}
$$
$$ or, h(w)= \begin{bmatrix}
h_1\\ h_2
\\ h_3
\\ ...
\\ ...
\\ h_n
\end{bmatrix}_{n\times 1} $$

If **y** is the output vector then error is given by

$$ 
error = 
\begin{bmatrix}
h_1\\ h_2
\\ h_3
\\ ...
\\ ...
\\ h_n
\end{bmatrix}-
\begin{bmatrix}
y_1\\ y_2
\\ y_3
\\ ...
\\ ...
\\ y_n
\end{bmatrix} $$

And error function / cost function in vectorized form is given by 
$$ J(W) = \frac{1}{2n}[(XW-y)^T(XW-y)]$$

In vectorized form gradient descent is given by (The whole coefficient vector can be updated at once)
$$ W = W - \frac{lr}{n}X^T(XW-y) $$

## Normal Equation
Normal equation is another way to find optimal coefficients of regression without using gradient descent algorithm.
Using normal eqation the optimal coefficient is given by 
$$ W = (X^T X)^{-1}X^Ty $$