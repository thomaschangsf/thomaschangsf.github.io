+++
title = "Essential Math For Data Science"
description = "A solid mathematics grasp helps one appreciate the beauty of ML."
+++

# 1 Basic Math And Calculus Review
### Exponential


### Logarithms
$$
log_2 8 = x

$$
- base = 2
- x = 3

	  - Benefits
		  - Logorithms simplifies operations, which helps computation training
			  - Multiplication: log(a x b) = log(a) + log(b)
			  - Division: log(a/b) = log(a) - log(b)
			  - Expoentiation log(a ^ n) = n * log(a)
			  - Inverse: log(x^-1) == log(1/x) = -log(x)
		  - Log is also good for feature engineering, transforming heavy tail distribution to a gaussian distribution, which is ideal for linear regression models

	  - Code
``` python
		from math import log
		
		# 2 raised to what power gives me 8?
		x = log(8, 2)
		# x = 3.0
```

### Euler's Number $e$ and Natural Logarithm
- $e$ = 2.71828 = euler number;
  - e is a constant like $\pi$
    - $e=(1 + 1/n)^n$
        - as n --> $\infty$, e approaches 2.71828
    - Application
        - To describe normal distribtuion
        - model logistic regression
        - Given equation to calculate compound interest, we can simplify it by using $e$
          $$A = P *(1 + r/n)^{nt} $$
          $$A=P * e^{rt}$$


- Natural logarithm
  - when use use e as the ==base== of our logarithm; $e$ is the default base for logarithm
  $$log_e(10) = ln(10)$$

```python
		from math import log, e
		
		# e raised to what power gives 10?
		x = log(10)
		x_detailed = log(10, e)
		
		#x = 2.3025
```


### Limit
- Limit express a value $\infty$ that is forever being approached, but never reached
  $$lim_{x \to \infty}1/x = 0$$


### Derivatives
- A derivative tells the slope of a function, and it is useful to measure the rate of change at any point in a function.
  $$ f(x) = x ^2 $$
  $$ \frac{\partial }{\partial x}f(x) = \frac{\partial}{\partial x}x^2 = 2x $$

```python
	def derivative_x(f, x, step_size):
	    m = (f(x + step_size) - f(x)) / ((x + step_size) - x)
	    return m
	
	def my_function(x):
	    return x**2
	
	derivative_x(my_function, 2, .00001) #4.0000100
```

- partial derivatives
    - Partial derivative enables use to find a slope with respect to multiple variables in several directions
        - Suppose u depends on 3 variables x, y, z.  To find the slope of u,
          $$\frac{\partial u}{\partial t}
          = h^2 \left( \frac{\partial^2 u}{\partial x^2}
        + \frac{\partial^2 u}{\partial y^2}
        + \frac{\partial^2 u}{\partial z^2} \right)$$


### Chain Rule
- Suppose we have 2 equations that are related like so
  $$ y = x^2 + 1$$
  $$z = y^3 - 2$$
  -To find $\frac{\partial}{\partial y}z$
  $$ \frac{\partial z}{\partial x} = \frac{\partial z} {\partial y} \frac{\partial y}{\partial x} $$

- Why do we care?
    - In neural network layers, chain rule enables se to "untangle" the derivative from each layer


### Integrals
- integrations find the area under a curve


# 2 Probability
### 2.1 Understanding Probability
- Statistics vs Probability
    - Statistics is the application of  probability to data
        - Ex: data distribtuion will impact what probability tool/approach we take



### 2.2 Probability Math
- Joint probability
    - what is the probability event_A ==and== event_B happen ==together== ?
    - P(A and B) = P(A) x P(B)

- Union probability
    - what is the probability event_A ==or== event_B happen?
    - P(A or B) = P(A) + P(B)

- Conditional Probability and Baye's Theorem
    - P (A | B) = P(B | A) * P(A)/ P(B)
    - Bayes theorem  can be used to chain several conditional probabilities together to keep updating our beliefs based on new information
    - Ex:
        - Given P(Cofee| Cancer) = 0.85, what is P(cancer | coffee?)
        - P(cancer|coffee) = P(coffee | cancer)* P(coffee) / P(cancer)
```python
	p_coffee_drinker = .65
	p_cancer = .005
	p_coffee_drinker_given_cancer = .85
	
	p_cancer_given_coffee_drinker = p_coffee_drinker_given_cancer *
	    p_cancer / p_coffee_drinker
```

- Joint and Union Conditional Probability
    - P(A and B) = P(A) x P(A|B)
    - P(A or B) = P(A) + P(B) - P(A|B) * P(B)



### 2.3 Binomial Probability Distribution
- Binomial distribution measures how likely ==k== successes can happen out of ==n== trials given we expect a ==p== probability
- Graph:
    - x-axis = num successes = k
    - y-axis = expected p probabilty
- Example
    - We run 10 tests, which either pass or fail (ie binomial) and has a success rate of 90%.  What is the probabibility of having 8 successes?
        - n=10; p=0.9; k=8
```python
	from scipy.stats import binom
	
	n = 10
	p = 0.9
	
	for k in range(n + 1):
	    probability = binom.pmf(k, n, p)
	    print("{0} - {1}".format(k, probability))
	
	# OUTPUT:	
	# 0 - 9.99999999999996e-11
	# 1 - 8.999999999999996e-09
	# 2 - 3.644999999999996e-07
	# 3 - 8.748000000000003e-06
	# 4 - 0.0001377809999999999
	# 5 - 0.0014880347999999988
	# 6 - 0.011160260999999996
	# 7 - 0.05739562800000001
	# 8 - 0.19371024449999993
	# 9 - 0.38742048900000037
	# 10 - 0.34867844010000004
```


### 2.4 Beta Probability Distribution
- Theta distribtion allows us to calculate the probability of a probability (yikes!)
    - In previous example, what is the probability that p=90%?  Beta distribution can help anser that question.
- Like all probability distribution, the area under the curve adds up to 1.
- graph
    - Given 8 successes and 2 failures
        - x-axis = p
        - y-axis = likelihood of p






# 3 Descriptive and Inferential Statistics
### 3.1 What is Data
- ==Data does not capture context or explanations==
    - data provides clues, and not truths
    - data may lead us to truth or mislead into erronous conclusions
        - ==I need to be curious how the data was created, who create it, and what the data is not capturing==
            - only 13% of ML projects succeeds



### 3.2 Descriptive vs Inferential Statistics
- Descriptive statistics summarizes the data
    - Ex: median, mean, , variance, standard deviation
- Inferential statistics attemps to uncover attributes of the ==larger== population based on ==a few samples==.
    - Ex: p-values, confidence interval, central limit theorem

### 3.3 Populations, Samples, and Bias
- A population is a particular group of interest we want to study
- A sample is a subset of the population that is ideally random and unbiased
- biases
    - different types
        - confirmation bias: capturing data that supports your belief
        - survival bias: caputre only the living subjects


### 3.4 Descriptive Statistics
- mean vs weighted mean
    - mean
        - over sample n:  $$\overline{x}=\sum_{i=0}^{n}\frac{x_i}{n}$$
        - over entire population N:
          $$\mu=\sum_{i=0}^{N}\frac{x_i}{N}$$
          $$\mu = \frac{x_1 + x_2 + .. + x_N}{N} $$

        - weighted mean
          $$ weightedmean =\frac{x_1*w_1 + x_2*w_2 + .. + x_n*w_n}{w_1 + w_2 + .. + w_n} 		$$

- Variance  vs standard deviation
    - variance : $$\sigma^2=\frac{\sum_{i=0}^{N}(x_i - \mu)^2}{N}$$
    - standard deviation = $$s^2=\frac{\sum_{i=0}^{n}(x_i - \overline{x})^2}{n-1} $$
- Normal distribution
    - Why normal distribution is useful
        -   It’s symmetrical; both sides are identically mirrored at the mean, which is the center.
        -   Most mass is at the center around the mean.
        -   It has a spread (being narrow or wide) that is specified by standard deviation.
        -   The “tails” are the least likely outcomes and approach zero infinitely but never touch zero
        -   It resembles a lot of phenomena in nature and daily life, and ==even generalizes nonnormal problems because of the central limit theorem==, which we will talk about shortly.
    - Probabibilty Density Function (PDF) vs Cumulative Density Function (CDF)
        - PDF for normal distribution $$f(x)=\frac{1}{\sigma}*(2*\pi)^{0.5}*e^{-\frac{1}{2}*(\frac{x-\mu^2}{\sigma})} $$
        - PDF returns ==liklihood== and NOT probabibility
            - Probability is the area under the PDF
            - CDF is the area of the PDF as the x axis value increases; so CDF can be used to return the probability for x < some value.
```python
		from scipy.stats import norm
		
		mean = 64.43 # mean weight of golden retreivers
		std_dev = 2.99
		x = norm.cdf(64.43, mean, std_dev)
		
		print(x) # prints 0.5; means 50% of the area, ie 
```

			- Inverse CDF (ppf)
				- There will be situations where we need to look up an area on the CDF and then return the corresponding x-value. This is a backward usage of the CDF
					- CDF: x-axis --> area/probability
					- Inverse CDF: area/probabibility --> x-axis
					- We use the ppf
					- Example:
						- I want to find the weight that 95% of golden retrievers fall under
```python
				from scipy.stats import norm
				
				x = norm.ppf(.95, loc=64.43, scale=2.99)
				print(x) # 69.3481123445849 lb
```




		- z score
			- use to rescale the normal distribution so the mean is 0 and the standard devation is 1. This makes it easy to compare the spread of one normal distribution to another normal distribution, even if they have different means and variances.



### 3.5 Inferential Statistics
- Central Limit Theorem
    -  As the population N size increases, even if that population does not follow a normal distribution, the normal distribution still makes an appearance. WOW.

- Confidence Interval
    - A confidence interval is a range calculation showing how confidently we believe a sample mean (or other parameter) falls in a range for the population mean.
        - Example:
          ```Based on a sample of 31 golden retrievers with a sample mean of 64.408 and a sample standard deviation of 2.05, I am 95% confident that the population mean lies between 63.686 and 65.1296 ```
    - IMOW:
        - you find the x value which encomposes 95% of the PDF area
    - Code
        1.  First find the critical z value $z_c$ which captures 95% of the PDF curve area
```python
			from scipy.stats import norm
		
			def critical_z_value(p):
			    norm_dist = norm(loc=0.0, scale=1.0)
			    left_tail_area = (1.0 - p) / 2.0
			    upper_area = 1.0 - ((1.0 - p) / 2.0)
			    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)
			
			print(critical_z_value(p=.95))
			# (-1.959963984540054, 1.959963984540054)
```


		     2. Use Central limit theorem to proce margein of error, which is the range around the sample mean that contains the population mean at that level of confidence. 

$$ E = \frac{+}{-} z_c * \frac{s}{n^{0.5}} $$

$$ 95\% confdence = \mu \frac{+}{} E $$

```python
			def confidence_interval(p, sample_mean, sample_std, n):
			    # Sample size must be greater than 30
			
			    lower, upper = critical_z_value(p)
			    lower_ci = lower * (sample_std / sqrt(n))
			    upper_ci = upper * (sample_std / sqrt(n))
			
			    return sample_mean + lower_ci, sample_mean + upper_ci
			
			print(confidence_interval(p=.95, sample_mean=64.408, sample_std=2.05, n=31))
			# (63.68635915701992, 65.12964084298008)
```

- Understanding P Values
    - p value is the probability of something occurring by chance rather than because of a hypothesized explanation
    - In the context of an experiment,
        - Given
            - our control variable, ie new model
            - the result of something happening, ie increase in click rate
        - A p value is low, ie < 5%,
            - means we ==discard== the $H_0$ null hypothesis
                - $H_0$ null hypotheseis states our variable has no impact; the result is random choice
            - means we accept the $H_1$ alternative hypothesis, which states the control variable is the reason behind the result.


- Hypothesis Testing
    - Example: does our pill (the variable) reduce the lenght of a fever?
        - Data
            - fever recovery is gaussian
                - mean = 18 days
                - std = 1.5
        - Experiment:
            - we give drug to 40 people and it took 16 days to recover. Did drug have impact?
        - One tail test
            - to be statistically significant, p < 0.05.  What is x, the number of day?
```python
				from scipy.stats import norm
				
				# Cold has 18 day mean recovery, 1.5 std dev
				mean = 18
				std_dev = 1.5
				
				# What x-value has 5% of area behind it?
				x = norm.ppf(.05, mean, std_dev)
				
				print(x) # 15.53271955957279
```

						- Ans= 15.5, which is less than our experiment mean recovery of 16 days, which means our pill is not statistically significant

### 3.6 T-Distribution: Dealing with Small Samples



### 3.7 Big Data Considerations and the Texas Sharpshooter Fallacy


# 4 Linear Algebra
### 4.1 What is a Vector?
- A vector is an arrow in space with a specific direction and length, often representing a piece of data
- Applications of vectors
    - Solvers like the one in Excel or Python PuLP use linear programming, which uses vectors to ==maximize a solution== while meeting those constraints
    - Computer graphics [manim library](https://docs.manim.community/en/stable/examples.html)
- Representation
    - v = [x, y, z]
        - implicit to the origin
    - code
```python
		import numpy as np
		v_nump = np.array([3, 2])
		
		v_python_list = [3, 2]
```
- Adding vectors
    - IMOW: the origin of the second vector is the end of the 1st vector
   ```python
      from numpy import array

      v = array([3,2])
      w = array([2,-1])
      
      # sum the vectors
      v_plus_w = v + w
      
      # display summed vector
      print(v_plus_w) # [5, 1]
```



- Scaling vectors
	- IMOW: scales the vector lengh 



- Span and Linear Dependence
	- ==Two vectors in opposite directions (ie one of their cordinate has opposite signs) can be used to create an inifinite spans through multiplication and adding==
	- Terminologies
		- span : the whole spae of possible vectors
		- linear indepdenent: two vectors in different directions (ie one of their coordinate has opposite sign)
		- linear depedent: two vectors in the same direction (or plane for multi-dim vector) 
	- So what?
		- Linear ==independent== vectors can create an infinite span
		- When solving system of equations, linear independent vectors gives us more "flexibility", which is important to solve the system of equations.
			- Conversely, linear dependent vectors are 
		- Later on, we will learn determinant, which is a tool to check for linear dependence 

### 4.2 Linear Transformations
- Basis Vectors $\hat{i}$ and $\hat{j}$
	- properties
		- lenght of 1
		- perpendicular to earch other
	- We can use $\hat{i}$ and $\hat{j}$  to create any vector we want by scaling and adding them.
	- Ex
``` python
		     i j
	basis = [1 0]
			[0 1]
```


### 4.3 Matrix Vector Multiplication
	- This transformation of a vector by applying basis vectors is known as _matrix vector multiplication_
``` python
			    i j
		[x'] = [a b] [x]
		[y'] = [c d] [y]
		
		[x'] = [ax + by]
		[y'] = [cx + dy]
		 
		from numpy import array
		# Declare i-hat and j-hat
		i_hat = array([2, 0])
		j_hat = array([0, 3])
		
		# compose basis matrix using i-hat and j-hat
		# also need to transpose rows into columns
		basis = array([i_hat, j_hat]).transpose()
		
		# declare vector v
		v = array([1,1])
		
		# create new vector
		# by transforming v with dot product
		new_v = basis.dot(v)
		
		print(new_v) # [2, 3]
```



### 4.4 Determinants
- During linar transformation, we can increase,decrease,rotate the original spacea area by X times.  The X times is the ==determinant==
```python
	from numpy.linalg import det
	from numpy import array
	
	i_hat = array([3, 0])
	j_hat = array([0, 2])
	
	basis = array([i_hat, j_hat]).transpose()
	
	determinant = det(basis)
	
	print(determinant) # prints 6.0; we increase area by 6.  Other transformations can rotate
```

### 4.5 Special Type of Matrices
- Square matrix:
    - numRows = numCols
    - application: eigendecomposition
- Identity matrix
    - diagonal are 1's; everything else is 0
    - application:
        - when you have an identity matrix, you essentially have undone a transformation and found your starting basis vectors  $\hat{i}$ and $\hat{j}$. This will play a big role in solving systems of equations in the next section.
- Inverse matrix
    - application: inverse matrix undoes the transformation of another matrix
    - $A^{-1}$ x $A^{1}$ = identity matrix
- Diagonal matrix
    - only diagonal has non-zero values
    - application: represents a simple scalar along the vector space
- Triangular matrix
    - only diagonal and the upper right diagonal have non-zero values
    - application: easier to solve in system of equations; good for decomposition tasks like [Lower Upper decompositison](https://en.wikipedia.org/wiki/LU_decomposition)
- Sparse matrix
    - matrix is mostly 0s
    - There is no interesting transformation, but this represents an opportunity to optimize the memory via more efficient representation.


### 4.6 System of Equations and Inverse Matrices
- Linear algebra is used to solve a system equation via inverse matrix
  - 
``` python
	Given                                            
		4x + 2y + 4z = 44
		5x + 3y + 7z = 56 
		9x + 3y + 6z = 71
	
	A = 4 2 4
	    5 3 7
		9 3 6
	
	B = 44
	    56
	    72
	
	AX = B
	X = (A^-1)(B) # !!! WOW
	
	A = Matrix([
	    [4, 2, 4],
	    [5, 3, 7],
	    [9, 3, 6]
	])
		
	from numpy import array
	from numpy.linalg import inv
	
	A = array([
	    [4, 2, 4],
	    [5, 3, 7],
	    [9, 3, 6]
	])
	
	B = array([
	    44,
	    56,
	    72
	])
	
	X = inv(A).dot(B)
	
	print(X) # [ 2. 34. -8.]

```


### 4.7 Eigenvectors and Eigenvalues
- Matrix decomposition is breaking up a matrix into its basic components, much like factoring numbers (e.g., 10 can be factored to 2 × 5).
    - There are multiple matrix decomposition techniques
        - Linear regression: QR decomposition (Chap 5)
        - eigendecomposition (ie used by PCA) (this chapter)
- In eigen decomposition, the original matrix A is decomposed to ==eigenvalue $\lambda$== and ==eigenvector== $\upsilon$
  $$A\upsilon =\lambda\upsilon$$

    - Example:
      $$
      A = 1 2
      4 5
      $$


$$	 
\lambda = eigenvalue = [-0.464, 6.464]
$$

$$ \upsilon = eigenvector = [ [0.0806, 0.0343], [0.59, -0.939]] $$

```python
					from numpy import array, diag
					from numpy.linalg import eig, inv
					
					A = array([
					    [1, 2],
					    [4, 5]
					])
					
					eigenvals, eigenvecs = eig(A)
					
					print("EIGENVALUES")
					print(eigenvals)
					print("\nEIGENVECTORS")
					print(eigenvecs)
					
					"""
					EIGENVALUES
					[-0.46410162  6.46410162]
					
					EIGENVECTORS
					[[-0.80689822 -0.34372377]
					 [ 0.59069049 -0.9390708 ]]
					"""
```



# 5 Linear Regression
### 5.1 Basic Linear Regression
```python
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.linear_model import LinearRegression
	
	# Import points
	df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
	
	# Extract input variables (all rows, all columns but last column)
	X = df.values[:, :-1]
	
	# Extract output column (all rows, last column)
	Y = df.values[:, -1]
	
	# Fit a line to the points
	fit = LinearRegression().fit(X, Y)
	
	# m = 1.7867224, b = -16.51923513
	m = fit.coef_.flatten()
	b = fit.intercept_.flatten()
	print("m = {0}".format(m))
	print("b = {0}".format(b))
	
	# show in chart
	plt.plot(X, Y, 'o') # scatterplot
	plt.plot(X, m*X+b) # line
	plt.show()
```


### 5.2 Residuals and Squared Errors
$$ residual_{squared} = (y_{predict} - y_{actual})^2$$
```python
	import pandas as pd
	
	# Import points
	points = pd.read_csv("https://bit.ly/2KF29Bd").itertuples()
	
	# Test with a given line
	m = 1.93939
	b = 4.73333
	
	sum_of_squares = 0.0
	
	# calculate sum of squares
	for p in points:
	    y_actual = p.y
	    y_predict = m*p.x + b
	    residual_squared = (y_predict - y_actual)**2
	    sum_of_squares += residual_squared
	
	print("sum of squares = {}".format(sum_of_squares))
	# sum of squares = 28.096969704500005
```

### 5.3 Finding the Best Fit Line

#### 5.3.1 closed form
- not computationally efficient
  $$m = \frac{n\sum{xy} - \sum{x}\sum{y}}{n\sum{x^2}-(\sum{x})^2} $$

$$b = \frac{\sum{y}}{n} - m\frac{\sum{x}}{n} $$


#### 5.3.2 Via Linear Algebra (Inverse Matrix Technique)
- Even though ML uses stochastic descent to solve the system of equations, it is instructional to see how linear algebra can be used to arrive at the close form solution
- more numerical stable
    - Numerical stability is how well an algorithm keeps errors minimized, rather than amplifying errors in approximations. Remember that computers work only to so many decimal places and have to approximate, so it becomes important our algorithms do not deteriorate with compounding errors in those approximations

$$b=(X^TX)^{-1} X^T y $$
```python
		import pandas as pd
		from numpy.linalg import inv
		import numpy as np
		
		# Import points
		df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
		
		# Extract input variables (all rows, all columns but last column)
		X = df.values[:, :-1].flatten()
		
		# Add placeholder "1" column to generate intercept
		X_1 = np.vstack([X, np.ones(len(X))]).T
		
		# Extract output column (all rows, last column)
		Y = df.values[:, -1]
		
		# Calculate coefficents for slope and intercept
		b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
		print(b) # [1.93939394, 4.73333333]
		
		# Predict against the y-values
		y_predict = X_1.dot(b)
```

- Enhance with QR Matrix Decomposition
    - As the matrix increase in dimension, we want to decompose our original matrix Z
      $$X = QR $$
      $$ b = R^{-1} Q^T y$$
```python
		import pandas as pd
		from numpy.linalg import qr, inv
		import numpy as np
		
		# Import points
		df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
		
		# Extract input variables (all rows, all columns but last column)
		X = df.values[:, :-1].flatten()
		
		# Add placeholder "1" column to generate intercept
		X_1 = np.vstack([X, np.ones(len(X))]).transpose()
		
		# Extract output column (all rows, last column)
		Y = df.values[:, -1]
		
		# calculate coefficents for slope and intercept
		# using QR decomposition
		Q, R = qr(X_1)
		b = inv(R).dot(Q.transpose()).dot(Y)
```


#### 5.3.3 Gradient Descent
- Gradient descent is an optimization technique that uses derivatives and iterations to minimize/maximize a set of parameters against an objective
    - so what is the objective?
        - when f(x) is linear equation, it's the minimization of the sum of squares. ==This residual squared is the graph you are trying to minimize.==
          $$ residual_{squared} = (y_{predict} - y_{actual})^2$$


- Ex: Gradient descet to find min of a parabola
  ```python
  import random
  
  def f(x):
      return (x - 3) ** 2 + 4
  
  def dx_f(x):
      return 2*(x - 3)
  
  # The learning rate
  L = 0.001
  
  # The number of iterations to perform gradient descent
  iterations = 100_000
  
   # start at a random x
  x = random.randint(-15,15)
  
  for i in range(iterations):
  
      # get slope
      d_x = dx_f(x)
  
      # update x by subtracting the (learning rate) * (slope)
      x -= L * d_x
  
  print(x, f(x)) # prints 2.999999999999889 4.0
  ```

- For linear equation, it's the minimization of the sum of squares
    - Ex: Gradient desceint for linear regression
  ```python
  import pandas as pd

  # Import points from CSV
  points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
  
  # Building the model
  m = 0.0
  b = 0.0
  
  # The learning Rate
  L = .001
  
  # The number of iterations
  iterations = 100_000
  
  n = float(len(points))  # Number of elements in X
  
  # Perform Gradient Descent
  for i in range(iterations):
  
      # TWC: Find the derivatives of our sum of squares function with respect to _m_ and _b_
      # slope of squared residual with respect to m
      D_m = sum(2 * p.x * ((m * p.x + b) - p.y) for p in points)
  
      # slope of squared residual with respect to b; 
      D_b = sum(2 * ((m * p.x + b) - p.y) for p in points)
  
      # update m and b
      m -= L * D_m
      b -= L * D_b
  
  print("y = {0}x + {1}".format(m, b))
  # y = 1.9393939393939548x + 4.733333333333227
    ```

### 5.4 Overfitting and Variance
-  The big-picture objective is not to minimize the sum of squares but to make accurate predictions on new data
	- If we fit a nonlinear curve to the data (overfitting), it will get 0 error, but probably will not generalize well to real data.
	- Overfitted model are more sensitive to outliers, thereby increase variance in our prediction.
- 
### 5.5 Stochasitc Gradient Descent

### 5.6 Correlation Coefficient

### 5.7 Statistical Significance

### 5.8 Conefficient of Determination

### 5.9 Standard Error of the Estimate

### 5.10 Prediction Intervals

### 5.11 Train/Test Splits

### 5.12 Multiple Linear Regressions


# 6 Logistic Regression and Classification
### 6.1 Understanding Logistic Regression
- A regression that predicts the probability of an outcome given 1+ independent variable.
- Characteristics
	- label = binary, or categorical # (multi-class)
	- features: $x_i \in X$
	- output = ==probability== -> can be converted into discrete value
	- fairly resilient to outliers

### 6.2 Performing Logistic Regression
- Logistic Function creates the sigmoid curve 
	- sigmoid curve transforms an input value x into an output range [0,1]
	- $$p(\mathbf{x} = \langle x_{1}, x_{2}, \dots, x_{n}\rangle) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}}$$
		- $\beta_0 + \beta_1$ are learned parameters
		- x is the input indpendent variable(s), feature(s)

	```python
		import math
	
		def predict_probability(x, b0, b1):
		    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
		    return p
	```


### 6.2.1 Finding the Coeffcients
- Previously, in linear regression, we used lease sqaures to find the linear equation.  
	- We can either used 
		- close form$$m = \frac{n\sum{xy} - \sum{x}\sum{y}}{n\sum{x^2}-(\sum{x})^2} $$

$$b = \frac{\sum{y}}{n} - m\frac{\sum{x}}{n} $$

		- inverse matrix 

$$b=(X^TX)^{-1} X^T y$$
 
		- approximation: (gradient descent)
- In logistic regression
	- we use _maximum likelihood estimation (MLE)_
		- maximizes the likelihood a given logistic curve would output the observed data (for binary, data label is 0,1)
		- MLE(logistic regression) vs least squared (linear regression)
	- We use gradient descent (or a library) to solve (there's no closed form)
		- sklearn library

```python
		import pandas as pd
		from sklearn.linear_model import LogisticRegression
		
		# Load the data
		df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")
		
		# Extract input variables (all rows, all columns but last column)
		X = df.values[:, :-1]
		
		# Extract output column (all rows, last column)
		Y = df.values[:, -1]
		
		# Perform logistic regression
		# Turn off penalty
		model = LogisticRegression(penalty='none')
		model.fit(X, Y)
		
		# print beta1
		print(model.coef_.flatten()) # 0.69267212
		
		# print beta0
		print(model.intercept_.flatten()) # -3.17576395
```

- MLE (details)
    - High level:
        - claculate parameters that bring our logistic curve to data points as closely as possible (vs regression: as close to the residual loss squared)
        - IMOW: calculate parameter to maximize the likihood probability to fit observed data to the sigmoid curve
    - For binary
      $$JointLiklihood = \prod_{i=1}^{n=totalSamples}(\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{y_i} (\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{1-y_i}$$

        - 1st term applies when data label is 1
        - 2nd term appleis when data label is 0
    - Mathematical optmization to prevent floating underflow: use log
        - multiple saller number gets you smaller number; ME235!
        - log transforms multiplication into addition
          $$JointLiklihood = \sum_{i=1}^{n=totalSamples}\log(\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{y_i} (\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{1-y_i})$$
- code1: Joint liklihood definition
```python
	joint_likelihood = Sum(log((1.0 / (1.0 + exp(-(b + m * x(i)))))**y(i) * \
		(1.0 - (1.0 / (1.0 + exp(-(b + m * x(i))))))**(1-y(i))), (i, 0, n))
```

- code2: use gradient descent (simpy)
``` python (simpy)
from sympy import *
import pandas as pd

points = list(pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples())

b1, b0, i, n = symbols('b1 b0 i n')
x, y = symbols('x y', cls=Function)
joint_likelihood = Sum(log((1.0 / (1.0 + exp(-(b0 + b1 * x(i))))) ** y(i) \
	* (1.0 - (1.0 / (1.0 + exp(-(b0 + b1 * x(i)))))) ** (1 - y(i))), (i, 0, n))

# Partial derivative for m, with points substituted
d_b1 = diff(joint_likelihood, b1) \
		   .subs(n, len(points) - 1).doit() \
		   .replace(x, lambda i: points[i].x) \
		   .replace(y, lambda i: points[i].y)

# Partial derivative for m, with points substituted
d_b0 = diff(joint_likelihood, b0) \
		   .subs(n, len(points) - 1).doit() \
		   .replace(x, lambda i: points[i].x) \
		   .replace(y, lambda i: points[i].y)

# compile using lambdify for faster computation
d_b1 = lambdify([b1, b0], d_b1)
d_b0 = lambdify([b1, b0], d_b0)

# Perform Gradient Descent
b1 = 0.01
b0 = 0.01
L = .01

for j in range(10_000):
    b1 += d_b1(b1, b0) * L
    b0 += d_b0(b1, b0) * L

print(b1, b0)
# 0.6926693075370812 -3.175751550409821
```

### 6.3 Multivariable Logistic Regression
- I frame the previous 6.2 section as logistic regression
- Example: Employment retention
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")

# grab independent variable columns
inputs = employee_data.iloc[:, :-1]

# grab dependent "did_quit" variable column
output = employee_data.iloc[:, -1]

# build logistic regression
fit = LogisticRegression(penalty='none').fit(inputs, output)

# Print coefficients:
print("COEFFICIENTS: {0}".format(fit.coef_.flatten()))
print("INTERCEPT: {0}".format(fit.intercept_.flatten()))

# Interact and test with new employee data
def predict_employee_will_stay(sex, age, promotions, years_employed):
    prediction = fit.predict([[sex, age, promotions, years_employed]])
    probabilities = fit.predict_proba([[sex, age, promotions, years_employed]])
    if prediction == [[1]]:
        return "WILL LEAVE: {0}".format(probabilities)
    else:
        return "WILL STAY: {0}".format(probabilities)

# Test a prediction
while True:
    n = input("Predict employee will stay or leave {sex},
        {age},{promotions},{years employed}: ")
    (sex, age, promotions, years_employed) = n.split(",")
    print(predict_employee_will_stay(int(sex), int(age), int(promotions),
          int(years_employed)))
```

### 6.4 Understanding the Log-Odds
- Log odds used by the logit function
- Since 1900s, mathematicians wants transform a linear function (==1 neural network layer==), and scale the output to the range of [0,1], corresponding to a probability
    - $$JointLiklihood = \sum_{i=1}^{n=totalSamples}\log(\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{y_i} (\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1+..+\beta_nx_n)}})^{1-y_i})$$
        - linear function: $\beta_0 + \beta_1x_1+..+\beta_nx_n$
        - ==log odds== = $e^{linear function} = e^{\beta_0 + \beta_1x_1+..+\beta_nx_n}$
            - logit function uses log odds; they are not the same
            - if we take the log of both side, $$log(\frac{p}{1-p}) = \beta_0 + \beta_1x_1+..+\beta_nx_n$$![[book-math-linear-to-log.png]]

- how to convert probability to odd?
    - logit function gives us probability p
    - $$odd = \frac{p}{1-p}$$
    - So in the employment retention example above,
        - if timeOfEmployment=6
            - then logit function returns  p = 72.7%
        - then $odd = \frac{0.727}{1-0.727}$ = 2.665
            - 2.66 times more likely to leave
    - More example: see diagram above


### 6.5 R-Squared
-$R^2$  indicates how well a given independent variable explains a dependent variable

$$R^2 = \frac{logLiklihood - logLiklihoodFit}{logLiklihood}$$
$$ logLiklihoodFit = \sum_{i=0}^n log(f(x_i)) \times y_i+log(1-f(x_i)) \times (1-y_i) $$
$$logLiklihood = \frac{\sum y_i}{n}\times y_i + (1-\frac{\sum y_i}{n})\times (1-y_i) $$
```python
import pandas as pd
from math import log, exp

patient_data = list(pd.read_csv('https://bit.ly/33ebs2R', delimiter=",") \
                                .itertuples())

# Declare fitted logistic regression
b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p

# calculate the log likelihood of the fit
log_likelihood_fit = sum(log(logistic_function(p.x)) * p.y +
                         log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
                         for p in patient_data)

# calculate the log likelihood without fit
likelihood = sum(p.y for p in patient_data) / len(patient_data)

log_likelihood = sum(log(likelihood) * p.y + log(1.0 - likelihood) * (1.0 - p.y) \
	for p in patient_data)

# calculate R-Square
r2 = (log_likelihood - log_likelihood_fit) / log_likelihood

print(r2)  # 0.306456105756576
```


### 6.6 P-Values
- We need to investigate how likely we would have seen this data by chance rather than because of an actual relationship. This means we need a p-value.

- Chi  Squared  distribution $\chi^2$
    - Degree of freedom = DOF = (# of parameters in our logistic regression - 1)
    - $\chi^2$ distribution with DOF = 1
        - sum each value in a normal distribution and saured
          $$ pValue = \chi^2(2(logLiklihoodFit) - logLiklihood)$$

```python
import pandas as pd
from math import log, exp
from scipy.stats import chi2

patient_data = list(pd.read_csv('https://bit.ly/33ebs2R', delimiter=",").itertuples())

# Declare fitted logistic regression
b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p

# calculate the log likelihood of the fit
log_likelihood_fit = sum(log(logistic_function(p.x)) * p.y +
                         log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
                         for p in patient_data)

# calculate the log likelihood without fit
likelihood = sum(p.y for p in patient_data) / len(patient_data)

log_likelihood = sum(log(likelihood) * p.y + log(1.0 - likelihood) * (1.0 - p.y) \
                     for p in patient_data)

# calculate p-value
chi2_input = 2 * (log_likelihood_fit - log_likelihood)
p_value = chi2.pdf(chi2_input, 1) # 1 degree of freedom (n - 1)

print(p_value)  # 0.0016604875618753787
```

### 6.7 Train/Test Splits
- Use k-fold validation to reuse all the data for training and test
- Ex: 3fold cross validation
```python
	import pandas as pd
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import KFold, cross_val_score
	
	# Load the data
	df = pd.read_csv("https://tinyurl.com/y6r7qjrp", delimiter=",")
	
	X = df.values[:, :-1]
	Y = df.values[:, -1]
	
	# "random_state" is the random seed, which we fix to 7
	kfold = KFold(n_splits=3, random_state=7, shuffle=True)
	model = LogisticRegression(penalty='none')
	results = cross_val_score(model, X, Y, cv=kfold)
	
	print("Accuracy Mean: %.3f (stdev=%.3f)" % (results.mean(), results.std()))
```

### 6.8 Confusion Matrices
						Actual:1       Actual:0
Predict: 1              TP                        FN                  Sensitivity

Predict: 0              FP                        TN                  Specificity

                         Precision                    Accuracy     


$$Precision=\frac{TP}{TP+FP}$$
$$Sensitivty=\frac{TP}{TP+FN}$$
$$Specificity=\frac{TN}{TN+FP}$$
$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$

$$F1Score=\frac{2\times Precision\times Recall}{Precision+ Recall}$$

```python
	import pandas as pd
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	from sklearn.model_selection import train_test_split
	
	# Load the data
	df = pd.read_csv('https://bit.ly/3cManTi', delimiter=",")
	
	# Extract input variables (all rows, all columns but last column)
	X = df.values[:, :-1]
	
	# Extract output column (all rows, last column)\
	Y = df.values[:, -1]
	
	model = LogisticRegression(solver='liblinear')
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33,
	    random_state=10)
	model.fit(X_train, Y_train)
	prediction = model.predict(X_test)
	
	"""
	The confusion matrix evaluates accuracy within each category.
	[[truepositives falsenegatives]
	 [falsepositives truenegatives]]
	
	The diagonal represents correct predictions,
	so we want those to be higher
	"""
	matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
	print(matrix)
```


### 6.9 Bayes Theorem and Classification
- One can use Bayes’ Theorem to bring in outside information to further validate findings on a confusion matrix

### 6.10 Receiver Operator Characteristics/Area Under Curve
- graph used tune the probability threshold to balance for the TPR vs FPR
    - y-axis = TPR = Sensitivity
    - x-axis = FPR = (1- Specificty)
- AUC is used to choose which model has better performance (tree vs linear regression)
- Code
```python
	results = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
	print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
	# AUC: 0.791 (0.051)
```

### 6.11 Class Imbalance
- Class imbalance is still an open problem with no great solution
- Possibilities
    - collect more data
    - try different models
    - confusion matrix and AUC to track
    -  generate synthetic samples of minority data (SMOTE algoritm)
    - duplication minority samples
  ```python 
	 	X, Y = ...
		X_train, X_test, Y_train, Y_test =  \
			train_test_split(X, Y, test_size=.33, stratify=Y)
	 
  ```


# 7 Neural Network
- A neural network is a multilayered ==regression== containing layers of weights, biases, and nonlinear functions that reside between input variables and output variables
-  Each node resembles a linear function before being passed to a nonlinear function (called an activation function)

### 7.1 When To Use Neural Network and Deep Learning
- Neural network and DL can be used for classification or regression
- Linear regression, logistic regression, and GBM trees do well on structured data
    - structure data can be represented as a table
- Use NN for unstructure data:
  - text: predict the next token
  - images: classify a text



### 7.2 Simple Neural Network
- node: a linear equation consiting of $$ X_1W_1 + X_2W_2+B_1 $$
    - where W are the weights, which are the arrows/path to the node
    - X is the input from the previous layer
- 1 layer = N nodes
- M layers
- So far, it looks similar to linear regression


#### 7.2.1 Activation Function
- An activation function is a nonlinear function that transforms or compresses the weighted and summed values in a node, helping the neural network separate the data effectively so it can be classified.
- Types of Activation function
  | Name      | Typical Layer | Description     | Notes     |
  | :---        |    :----:   |          ---: |          ---: |
  | Logistic      | Output        | S-shaped sigmoid curve   | output in [0,1]; used in binary classifcation |
  | Softmax      | Output        | Ensures all outputs nodes sumes to 1   | Useful for multiple classifcations and scale outputs add to 1.0|
  | Tangent Hyperbolic      | Output        | tanh, S-shaped sigmoid curve between -1 and 1    | Assists in “centering” data by bringing mean close to 0 |
  | Relu      | Hidden       | Turns negative values to 0    | Popular activation faster than sigmoid and tanh, mitigates vanishing gradient problems and computationally cheap |
  | Leaky Relu      | Hidden       | Multiples neg values by 0.01    | Controversial variant of ReLU that marginalizes rather than eliminates negative values|


#### 7.2.2 Forward Propagation
- Mental model with L layers
    - input X --> Hidden: $Relu( W @ X + B)_{layer-1}$ --> output: $logistic(W@X + B)_{layer_L}$

```python
	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	
	all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")
	
	# Extract the input columns, scale down by 255
	all_inputs = (all_data.iloc[:, 0:3].values / 255.0)
	all_outputs = all_data.iloc[:, -1].values
	
	# Split train and test data sets
	X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
	    test_size=1/3)
	n = X_train.shape[0] # number of training records
	
	# Build neural network with weights and biases
	# with random initialization
	w_hidden = np.random.rand(3, 3)
	w_output = np.random.rand(1, 3)
	
	b_hidden = np.random.rand(3, 1)
	b_output = np.random.rand(1, 1)
	
	# Activation functions
	relu = lambda x: np.maximum(x, 0)
	logistic = lambda x: 1 / (1 + np.exp(-x))
	
	# Runs inputs through the neural network to get predicted outputs
	def forward_prop(X):
	    Z1 = w_hidden @ X + b_hidden
	    A1 = relu(Z1)
	    Z2 = w_output @ A1 + b_output
	    A2 = logistic(Z2)
	    return Z1, A1, Z2, A2
	
	# Calculate accuracy
	test_predictions = forward_prop(X_test.transpose())[3] # grab only output layer, A2
	test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
	accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
	print("ACCURACY: ", accuracy)
```

### 7.3 Backpropagation
#### 7.3.1 Calculate the Weight and Bias Derivatives
- For each layer, we do a partial derivative of the cost function
    - $C=(A_{layer_i-1}^2-Y_{layer_i-1})$
        - C is the cost function
        - $A_{layer-1}$ is the output of layer i-1
            - = ReLU($Z_{layer-1} = (W @ X + B)_{layer-2}$)


#### 7.3.2 Stochastic Gradient Descent
Raw
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Learning rate controls how slowly we approach a solution
# Make it too small, it will take too long to run.
# Make it too big, it will likely overshoot and miss the solution.
L = 0.05

# Extract the input columns, scale down by 255
all_inputs = (all_data.iloc[:, 0:3].values / 255.0)
all_outputs = all_data.iloc[:, -1].values

# Split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1 / 3)
n = X_train.shape[0]


# Build neural network with weights and biases
# with random initialization
w_hidden = np.random.rand(3, 3)
w_output = np.random.rand(1, 3)

b_hidden = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

# Activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# Derivatives of Activation functions
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# returns slopes for weights and biases
# using chain rule
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dC_dA2 = 2 * A2 - 2 * Y
    dA2_dZ2 = d_logistic(Z2)
    dZ2_dA1 = w_output
    dZ2_dW2 = A1
    dZ2_dB2 = 1
    dA1_dZ1 = d_relu(Z1)
    dZ1_dW1 = X
    dZ1_dB1 = 1

    dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T

    dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2

    dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1

    dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T

    dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# Execute gradient descent
for i in range(100_000):
    # randomly select one of the training data
    idx = np.random.choice(n, 1, replace=False)
    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    # run randomly selected training data through neural network
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # distribute error through backpropagation
    # and return slopes for weights and biases
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # update weights and biases
    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

# Calculate accuracy
test_predictions = forward_prop(X_test.transpose())[3]  # grab only A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("ACCURACY: ", accuracy)
```


- Using scikit
```python
import pandas as pd
# load data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('https://bit.ly/3GsNzGt', delimiter=",")

# Extract input variables (all rows, all columns but last column)
# Note we should do some linear scaling here
X = (df.values[:, :-1] / 255.0)

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Separate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(3, ),
                   activation='relu',
                   max_iter=100_000,
                   learning_rate_init=.05)

nn.fit(X_train, Y_train)

# Print weights and biases
print(nn.coefs_ )
print(nn.intercepts_)

print("Training set score: %f" % nn.score(X_train, Y_train))
print("Test set score: %f" % nn.score(X_test, Y_test))
```



### 7.4 Limitation of Neural Networks and Deep Learning
- NN can overfit
    - layers, nodes, and activation functions makes it flexible fitting to data in a nonlinear manner



### 7.5 Conclusion



# 8 Career Advice and Path Forward
- data science is software engineering with proficiency in statistics, machine learning, and optimization

