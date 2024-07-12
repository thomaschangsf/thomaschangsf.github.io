+++
title = "DataScience From Scratch"
description = "Great book on how to implement data science from first principles."
+++

# 1 Introduction


# 2 Crash Course In Python

_Tuples_
- List's immutable cousin
```python
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3      # my_list is now [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")
```

- convenient way to return multiple values and multiple assignment
```python
def sum_and_product(x, y):
    return (x + y), (x * y)
sp = sum_and_product(2, 3)   

x, y = 1, 2     # now x is 1, y is 2
x, y = y, x     # Pythonic way to swap variables; now x is 2, y is 1

```

_Dict_
```python
# Literal assignment
grades = {"Joel": 80, "Tim": 95}    # dictionary literal

# Access non-existent key will throw KeyError exception
grade["Kate"]

# Better way; 
# defaults to None
grades.get("Kate") # or grades.get("Kate", 0)

```



_DefaultDict_
- Handling non-existent keys is cumbersome; defaultdict can help for various types, such as int, list.

```python
from collections import defaultdict

word_counts = defaultdict(int)          # int() produces 0
for word in document:
    word_counts[word] += 1


dd_list = defaultdict(list)             # list() produces an empty list
dd_list[2].append(1)                    # now dd_list contains {2: [1]}

dd_dict = defaultdict(dict)             # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle"     # {"Joel" : {"City": Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1                       # now dd_pair contains {2: [0, 1]}
```


_Counters_
- Turns a sequence of values into a defauldict(int)
```python

from collections import Counter
c = Counter([0, 1, 2, 0])          # c is (basically) {0: 2, 1: 1, 2:1}

# recall, document is a list of words
word_counts = Counter(document)
word_counts.most_common(10) # returns top 10 common word
```

_Set_
- if you want fast lookup time, use a set and not a tuple
```python
primes_below_10 = {2,3,5,7}
```

_Truthiness_
```python
# None indicates nonexistent value
x = {}.get("Something")
assert x is None # more pythonic than x == None

# Empty containers and values are Falsy
# [], {}, set, 0, 

```


_Sorting a List_
- 2 methods
    - sort method sorts in place
    - sorted returns a new list

```python
x = [4, 1, 2, 3]
y = sorted(x)     # y is [1, 2, 3, 4], x is unchanged
x.sort()          # now x is [1, 2, 3, 4]
```

- By default, sort in ascending order. To reverse, we have two options
```python
# sort the list by absolute value from largest to smallest
x = sorted([-4, 1, -2, 3], key=abs, reverse=True)  # is [-4, 3, -2, 1]

# sort the words and counts from highest count to lowest
wc = sorted(word_counts.items(),
            key=lambda word_and_count: word_and_count[1],
            reverse=True)
```


_Generators_
- Generators are useful when you need to access 1 element at a time; unlike lists where all the elements are stored in memory
```python
def generate_range(n):
    i = 0
    while i < n:
        yield i   # every call to yield produces a value of the generator
        i += 1

for i in generate_range(10):
    print(f"i: {i}")
```
- Iterable signifies it's lazy
```python
    evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
```

_Regular Expressions_
- Fairly complex and powerful module for expression matching
    - [doc](https://docs.python.org/3/library/re.html)
- re.match checks whether the beginning of a string matches a regular expression
- re.search checks whether any part of a string matches a regular expression.
- Example
```python
import re

re_examples = [                        # All of these are True, because
    not re.match("a", "cat"),              #  'cat' doesn't start with 'a'
    re.search("a", "cat"),                 #  'cat' has an 'a' in it
    not re.search("c", "dog"),             #  'dog' doesn't have a 'c' in it.
    3 == len(re.split("[ab]", "carbs")),   #  Split on a or b to ['c','r','s'].
    "R-D-" == re.sub("[0-9]", "-", "R2D2") #  Replace digits with dashes.
    ]

assert all(re_examples), "all the regex examples should be True"
```


_args and kwargs_
- args and kwargs enables one to specify a function that takes arbitrary arguments
    - args is a tuple of its unnamed arguments
    - kwargs is a dict of its named arguments

- Simple example
```python
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

# prints
#  unnamed args: (1, 2)
#  keyword args: {'key': 'word', 'key2': 'word2'}
```


- More complex function with high order function (function that takes a function as its input)
```python
def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through to f"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"
```


_Type Annotation_
- Python is dynamically typed;  types operations are loosely enforced as long as it runs
```python
def add(a, b):
    return a + b

assert add(10, 5) == 15,                  "+ is valid for numbers"
assert add([1, 2], [3]) == [1, 2, 3],     "+ is valid for lists"
assert add("hi ", "there") == "hi there", "+ is valid for strings"

try:
    add(10, "five")
except TypeError:
    print("cannot add an int to a string")
```

- While python does not enforce strict type, type annotation in python has these benefits
    - types are a form of documentation
    - External tools (ie mypy) leverages types
        - autocompletion
    - Thinking about types may help/highlight need for simpler design
  ```python
  # Hmm.. is operation too wide?  we are allowing so many different operators
  def ugly_function(value: int,
                operation: Union[str, int, float, bool]) -> int:
  ```

- Some Types in typing module
    - Dict, Iterable, Tuple
  ```python
  from typing import Dict, Iterable, Tuple
  # keys are strings, values are ints
  counts: Dict[str, int] = {'data': 1, 'science': 2}
  # lists and generators are both iterable
  if lazy:
      evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
  else:
      evens = [0, 2, 4, 6, 8]
  # tuples specify a type for each element
  triple: Tuple[int, float, int] = (10, 2.3, 5)
  
  ```

    - Optional
  ```python
  from typing import Optional
  best_so_far: Optional[float] = None  # allowed to be either a float or None
  ```

    - Callable is a type annotation for function
  ```python
  from typing import Callable
  
  # The type hint says that repeater is a function that takes
  # two arguments, a string and an int, and returns a string.
  def twice(repeater: Callable[[str, int], str], s: str) -> str:
      return repeater(s, 2)
  
  def comma_repeater(s: str, n: int) -> str:
      n_copies = [s for _ in range(n)]
      return ', '.join(n_copies)
  
  assert twice(comma_repeater, "type hints") == "type hints, type hints"
  ```



# 3  Visualizing Data
### 3.1 matplotlib
- matplotlib python module is good for basic visualizations (bar, line, scatter). For more interactive alternatives, see section 5.
```python
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# add a title
plt.title("Nominal GDP")

# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()
```


### 3.2 Bar Charts
_Basic Example_
```python
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# plot bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")     # add a title
plt.ylabel("# of Academy Awards")   # label the y-axis

# label x-axis with movie names at bar centers
plt.xticks(range(len(movies)), movies)

plt.show()
```

_Histograms: Buckets _
```python
from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Bucket grades by decile, but put 100 in with the 90s
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # Shift bars right by 5
        histogram.values(),                 # Give each bar its correct height
        10,                                 # Give each bar a width of 10
        edgecolor=(0, 0, 0))                # Black edges for each bar

plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105,
                                           # y-axis from 0 to 5

plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
```

### 3.3 Line Charts
- Good for showing trends
- Example: Bias vs Variance Tradeoff
    - x-axis: model complexity
```python
# Each index in the 3 lists below corresponds to a model complexity
variance     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error  = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# We can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.plot(xs, variance,     'g-',  label='variance')    # green solid line
plt.plot(xs, bias_squared, 'r-.', label='bias^2')      # red dot-dashed line
plt.plot(xs, total_error,  'b:',  label='total error') # blue dotted line

# Because we've assigned labels to each series,
# we can get a legend for free (loc=9 means "top center")
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()
```



### 3.4 Scatter Plots
- Good to visualizing ==relationship== between 2 sets of data
```python
friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy=(friend_count, minute_count), # Put the label with its point
        xytext=(5, -5),                  # but slightly offset
        textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()
```


### 3.5 Further Exploration
- The matplotlib [Gallery](https://matplotlib.org/2.0.2/gallery.html) will give you a good idea of the sorts of things you can do with matplotlib (and how to do them).
- seaborn is built on top of matplotlib and allows you to easily produce prettier (and more complex) visualizations.
- Altair is a newer Python library for creating declarative visualizations.
- D3.js is a JavaScript library for producing sophisticated interactive visualizations for the web. Although it is not in Python, it is widely used, and it is well worth your while to be familiar with it.
- Bokeh is a library that brings D3-style visualizations into Python.


# 4 Linear Algebra
### 4.1 Vectors
- Vector are objects that
    - can be added with other vectors
    - multiplied by a scalar to form another vector
- Defining our Vector type alias
```python
	from typing import List
	Vector = List[float]
```
- Common vector operators
```python
	# Add 2 vectors 
	def add(v: Vector, w: Vector) -> Vector:
	    """Adds corresponding elements"""
	    assert len(v) == len(w), "vectors must be the same length"
	    return [v_i + w_i for v_i, w_i in zip(v, w)]	
	assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

	# Scale a vector by a scalar
	def scalar_multiply(scalar: float, v: Vector) -> Vector:
	    return [scalar * v_i for v_i in v]
	assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

	# Dot product measures the similarity of 2 vectors based on the direction they are pointing to.
	# dot_exact(a,b) = |a| [b] cos(theta)
	# dot_approx = sum(a_i * b_i + a_j * b_j ...)
	#    WOW: don't need to know the angle theta
	def dot(v: Vector, w: Vector) -> float:
		return sum(v_i * w_i for v_i, w_i in zip(v, v))
	assert dot([1, 2, 3], [4, 5, 6]) == 32


	# sum of sqaures of 1 vector can be defined in terms of dot function
	def sum_of_squares(v: Vector) -> float:
		return dot(v, v)
	assert sum_of_squares([1, 2, 3]) == 14

	# mangnitude of a vector can be defined using sum_of_squares
	# aka L2 Norm
	def magnitude(v: Vector) -> float:
		return math.sqrt(sum_of_sqaures(v))
	assert magnitude([3, 4]) == 5

```

- Distance between two vectors (Euclidean)
  $$ \sqrt{(v_1-w_1)^2 + .. + (v_n-w_n)^2}$$
```python
	def squared_distance(v: Vector, w: Vector) -> float:
	    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
	    return sum_of_squares(subtract(v, w))
	
	def distance(v: Vector, w: Vector) -> float:
	    """Computes the distance between v and w"""
	    return math.sqrt(squared_distance(v, w))
```

-
    - other distance measurements [blog](https://machinelearningmastery.com/distance-measures-for-machine-learning/#:~:text=Euclidean%20distance%20is%20calculated%20as,differences%20between%20the%20two%20vectors.&text=If%20the%20distance%20calculation%20is,to%20speed%20up%20the%20calculation.)
        - L2 norm calculates the distance of ==1== vector from the origin
        - Manhattan distance - shortest path between 2 points on the grid
      ```python
      def manhattan_distance(a, b):
          return sum(abs(e1-e2) for e1, e2 in zip(a,b))
      ```
        - Minkowski distance: generalization of euclidean and manhattan




### 4.2 Matrices
- In practice, use numpy for better performance. But we will implement our own for better understanding
- Type aliases
``` python
# Another type alias
Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],     # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]
```

- Matrix accessor
```python
from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]             # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j]          # jth element of row A_i
            for A_i in A]   # for each row A_i

```

- Matrix creation
```python
from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list
             for j in range(num_cols)]  #   [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)
```


### 4.3 Further Exploration


# 5 Statistics
### 5.1 Describing a Single Set of Data
_Central Tendencies_
- A common metric is to quantify where our data is centered (mean and median)
```python
def mean(xs: List[float]) -> float:
	return sum(xs)/len(xs)


# median
# if odd length --> mid element; if even length --> avg of the middel two elements
def _median_odd(xs: List[float]) -> float:
	mid_index = len(xs)//2
	return sorted(xs)[mid_index]

def _median_even(xs: List[float]) -> float:
	sorted_xs = sorted(xs)
	high_index = len(xs)//2
	return (sorted_xs[high_index-1] + sorted_xs[high_index]) /2

def median(xs: List[float]) -> float:
	return _median_even(xs) if len(xs)%2 == 0 else _median_odd(xs)

```


- Percentile: generalization of median
```python
def quantile(xs: List[float], percentile: float) -> float:
	index = int(percentile * len(xs))
	return sorted(xs)[p_index]
	
```

- Mode : most common values
```python
def mode(xs: List[float]) -> List[float]:
	counts = Counter(x)
	max_count = max(counts.values())
	return [ x_i for x_i, count in counts.items() if count == max_count ]
```


_Dispersion_
- Dispersion measures how spread out our data is

```python
# CON: does not use the entire data set
def data_range(xs: List[float]) -> float:
	return max(xs) - min(xs)

def variance(xs: List[float]) -> float:
	xs_mean = mean(xs)
	deviations = [x - xs_mean for x in xs]
	return dot(deviations, deviations) / (len(xs) - 1)

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))

										  
```


### 5.2 Correlation
- Correlation measures the relationship between two arrays

_Covariance_
- Covariance vs variance
    - Variance measures how variable deviate from the mean
    - Covariance measures how 2 variables vary in tandem from their respective means

```python
def covariance(xs: List[float], ys: List[Float]) -> float:
	""" +1 -> pos relationship; 0 no relationship """
	assert len(xs) == len(ys), "xs and ys must be same length"

	def mean_diff(ls: List[float]) -> List[float]:
		mean_ls = sum(ls)/len(ls)
		return [i-mean_ls for i in ls]

	return dot(mean_diff(xs), mean_diff(ys)) / (len(xs) -1)
	
```


- Correlation
    - Covariance is hard to interpret because each input is not normalized to standard deviation of each inputs;
```python
def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0    # if no variation, correlation is zero
```


### 5.3 Simpson's Paradox
- Correlation may be misleading
    - Correlation assume ==all else being equal==
    - But factors like
        - how we sample/construct the dataset can affect the correlation result significantly



### 5.4 Other Correlation Caveats

### 5.5 Correlation and Causation
- Correlation is not causation

### 5.6 Further Exploration



# 6 Probability
### 6.1 Dependence and Independence
- 2 events E and F are dependent if knowing something about E gives us information whether F happens; otherwise they are independent.
- If events E and F are independent, then probability of both event E and F happening
  $$ P(E,F) = P(E) P(F)$$

- Examples
    - probability of rolling 2 dices is 1/2 * 1/2

### 6.2 Conditional Probability
- If two events are dependent
  $$P(E | F) = P(E,F)/P(F)$$

### 6.3 Baye's Theorem
- Think of it as a way to reverse conditional probabilities
  $$P(E | F) = P(E,F)/P(F)$$
  $$P(E|F) = F(F|E) P(E) / P(F)$$


### 6.4 Random Variables
- Random variable have values which has a probability distribution.
- Expected value: the average of a random's variable weighted by its probability
    - Ex: Coin toss where head=0 and tail=1

$$Expected Value = value_{head} * prob_{head} + value_{tail} * prob_{tail} $$
$$Expected Value = 0 * 0.5 + 1 * 0.5 $$


### 6.5 Continuous Distribution
- Random variables value can follow a
    - discrete distribution:  dices rolls: {1,2,3,4,5,6}
    - continuous distribution - values falls within a continuous spectrum
        - calculus integration:
            - if value follows a density function f
            - then probability of seeing a value between x and x+h is  h* f(x) for small h

- Cumulative distribution function (CDF)
    - gives probability variable is less than or equal to a value


### 6.6 Normal Distribution
- Normal distribution is a particular continuous distribution described by its mean $\mu$ and standard deviation $\sigma$

$$f(x|\sigma, \mu) = \frac{1}{\sqrt{2\pi\sigma}}exp(-\frac{(x-\mu^2)}{2\sigma^2})$$
```python
	import math
	SQRT_TWO_PI = math.sqrt(2 * math.pi)
	
	def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))
```

_Probability Distribution Function PDF_
```python
	import matplotlib.pyplot as plt
	xs = [x / 10.0 for x in range(-50, 50)]
	plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
	plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
	plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
	plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
	plt.legend()
	plt.title("Various Normal pdfs")
	plt.show()
```

_CDFs for Normal Distribution_ using the erf function
```python
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

# PLOTS for normal CDF for various values for mean and deviation
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()
```


- Use CDF to find value ($\mu \sigma$?) corresponding to a certain probability
```python
def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
    """Find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0                      # normal_cdf(-10) is (very close to) 0
    hi_z  =  10.0                      # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # Consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the CDF's value there
        if mid_p < p:
            low_z = mid_z              # Midpoint too low, search above it
        else:
            hi_z = mid_z               # Midpoint too high, search below it

    return mid_z
```


### 6.7 Central Limit Theorem
- a random variable defined as the average of a ==large number of independent and identically distributed random variables== is itself approximately normally distributed.

- Ex: Bernouli distribution with large enough sample size follows a normal distribution
    - Bernouli:
        - mean of Bernouli is p
        - standard deviation is $\sqrt{p(1-p)}$
    - Large sample --> follows normal distribution
        - mean = np
        - standard deviation = $\sqrt{np(1-p)}$

```python
def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n, p) and plots their histogram"""
    data = [binomial(n, p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()
```



### 6.8 Further Exploration



# 7 Hypothesis and Inference
### 7.1  Statistical Hypothesis Testing
- Terminologies
    - $H_0$ : aka null hypothesis, represent the default position
        - Type 1 error (false positive) : wrongly reject $H_0$ even though it is true
            - Significance : how willing we are to make a type 1 error (usually 5% or 1%)
        - Type 2 error (false negative) : fail to reject $H_0$ even though it's false
            - power of test:
    - $H_1$ : alternative hypothesis

- We use statistics to decide whether we can reject $H_0$
    - The different approaches below differ on the what the probability is on, ie parameters like normal distribution $\mu, \sigma$, value, etc..

- Approach 1: Choose bounds based on some probability cutoff using the CDF
    - See 7.2 below




### 7.2 Example: Flipping a Coin
- Overview
    - Our default hypothesis: probability of head is 0.5
        - What is the range of  head counts we need to observe to satisfy our requirements on
            - significance: type 1 error
            - power: type 2 error?
    1.  Model probability of head in coin toss as Bernoulli
    2. Central limit theorem --> Bernoulli distribution becomes normal distribution
    3. Use CDF to calculate various count_head needs to be observed


- Null hypothesis : probability of head is 0.5 (p_head = 0.5)

- Model random variable's value as binomial(n=#numSamples, p=probability_head)

- If large enough sample, binomial distribution can be approximated as normal distribution, and characterized by $\mu, \sigma$
```python
from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
```

- Since we assume random variable follows normal distribution, we can use norml_cdf to quantify the probability that the value (p_head or ($\mu, \sigma)$?) lies within or outside a particular interval
```python
from scratch.probability import normal_cdf

# The normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is not between lo and hi."""
    return 1 - normal_probability_between(lo, hi, mu, sigma)
```

- Similarly, we can find either the nontail region or the (symmetric) interval around the mean that accounts for a certain level of likelihood
    - Ex: we want to find an interval centered at the mean and containing 60% probability, then we find the cutoffs where the upper and lower tails each contain 20% of the probability (leaving 60%)
```python
from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    Returns the symmetric (about the mean) bounds
    that contain the specified probability
    """
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound
```

- If our hypothesis is correct, then $\mu=500, $\sigma=15.8$
  ``` python
  mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
  
  ```

- Signficance: What's our appetite for false positive (Type 1 error)?
    - Let's choose significance (ie willingness to make a false positive) to be 5% --> p=0.95
  ```python
  # (469, 531)
  lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
  
  ```

    - In order for us to reject H0 with 5% chance of false positive, we need to see between 469 and 531 heads

- Power of Test : What's our appetite for false negative (Type 2 error)
    - this quantifies our probability of a type 2 error
```python
	# 95% bounds based on assumption p is 0.5
	#(469, 531)
	lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
	
	# actual mu and sigma based on p = 0.55
	mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
	
	# a type 2 error means we fail to reject the null hypothesis,
	# which will happen when X is still in our original interval
	type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
	power = 1 - type_2_probability      # 0.887
	```
	
 - 
	- We reject H_0 when we see 


### 7.3 Approach 2: p-Values
- We compute the probability, assuming $H_0$ is true, that we see a value as extreme as the one we actually observed.

```python
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma)?
    """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)
```

- If we were to see 530 heads, our p value is 0.062; so our $H_0$ hypothesis is valid



### 7.4 Approach 3: Confidence Intervals
- Quantify the confidence interval around the observed value of the parameter, ie probability of head in a coin toss.
- Ex:
    - If we observe 525 heads out of 1000 flips, --> p_head = 0.525.
    - What is our confidence level that p_head is actually 0.525?
      ```python
      # We use our observation, and assign p_hat = 0.525
      p_hat = 0.525
      sigma = math.sqrt(p_hat * (1-p_hat) #0.0158

      # probability threadhold where we are 95% confident is the true p
      p_lower, p_higher = normal_two_sided_bounds(0.95, mu, sigma)        
      # [0.4940, 0.5560]

      # Since H_0 has p=0.5, then H_0 falls within our 95% confidence interval
      ```


### 7.5 p-Hacking


### 7.6 Example: Running on A/B Test
- Given 2 Ads A and B
    - where
        - $n_A$ = number of users we clicked on Ad A
        - $N_A$= total number of ad A shown
        - $n_B$ = number of users we clicked on Ad B
        - $N_B$= total number of ad B shown
    - Our null hypothesis $H_0$ is the original ad A is better
    - We want to know if we should roll out A or B

- Click is a Bernouli distribution, but with enough sample, it follows a normal distribution which can be characterized by $\mu , \sigma$
```python
	def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
	    p = n / N
	    sigma = math.sqrt(p * (1 - p) / N)
	    return p, sigma
	```


- Calculate the difference of $\mu , \sigma$
	- The 2 normal distribution A and B are independent
		- $p_{diff} = p_a - p_b$
		- $\sigma_{diff} = \sqrt{\sigma_a^2 + \sigma_b^2}$
- Our Null hypothesis is $p_A$ and $p_B$ are the same
```python
	# returns the z value
	def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
	    p_A, sigma_A = estimated_parameters(N_A, n_A)
	    p_B, sigma_B = estimated_parameters(N_B, n_B)
	    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
```


- Ex 1: Insufficient to disprove null hypothesis
```python
z = a_b_test_statistic(1000, 200, 1000, 180)    # -1.14
two_sided_p_value(z)                            # 0.254
```

- Ex 2: Sufficient to disprove null hypothesis
```python
z = a_b_test_statistic(1000, 200, 1000, 150)    # -2.94
two_sided_p_value(z)                            # 0.003
```




### 7.7 Approach 4: Bayesian Inference
- Rather than making probability judgments about the tests, you make probability judgments about the parameters.
    - Assume a prior distribution; collect data ; update the posterior


### 7.8 Further Exploration



# 8 Gradient Descent
- In ML, training a model often means finding the best parameters.
    - minimization problem: minimize the error of its prediction
    - maximization problem: maximize the likelihood of the data
- Gradient gives the input direction which the function increases most quickly
    - maximization -> gradient
    - minimization -> - gradient

### 8.1 Estimating Gradient

- Defining Scalar Gradient
``` python
from typing import Callable

# returns gradient
def gradient_scalar(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

# Funtion we are trying to calculate the derivative
def square(x: float) -> float:
    return x * x

# Test
xs = range(-10, 11)
estimates = [gradient_scalar(square, x, h=0.001) for x in xs]

# plot to show they're basically the same --> shows a line with a slope of 2
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, estimates, 'b+', label='Estimate')   # blue +
plt.legend(loc=9)
plt.show()
```


- When the function f has many variables (ie input is a vector), each element has a gradient; it is a partial derivative
```python
def gradient_vector(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [gradient_vector(f, v, i, h)
            for i in range(len(v))]
```


### Using the Gradient
```python
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]

# let f be the function we are trying to optimize
for epoch in range(1000):
	# compute the gradient at v
    grad = estimate_gradient(lambda x: x * x , v, step_size=0.001)    
    v = gradient_step(v, grad, -0.01)    # take a negative gradient step
    print(epoch, v)

```

### Choosing the Right Step Size
- Some options are:
    - Use a fixed step size
    - Gradually shrink the step size over time
    - At each step, choose the step size that minimizes the value of the objective function

### Using Gradient Descent to Fit Models
- Let's assume we have a linear model where we are trying to learn the parameter theta, (ie slope and intercept)
```python
# Method clculates the error between prediction and actual
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta.            # theta is the parameter vector we are learning
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual).
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad
```

- Code steps through
    1. Start with a random value for theta.
    2. Compute the mean of the gradients.
    3. Adjust theta in that direction.
    4. Repeat.
```python
from scratch.linear_algebra import vector_mean

# Start with random values for slope and intercept
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    # Compute the mean of the gradients
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
```

### Mini-batch and Stochastic Gradient Descent
- mini-batch gradient descent: calculate gradient descent using only a subset of larger dataset:
- stochastic gradient descent: calculate gradient descent using only ==ONE== sample:


# 9 Getting Data
### 9.1 stdin and stdout
- We can use unix pipe with python files to create interesting functionality
    - pipes takes the output from the left cmd as input to the next output command

- Example 1: Pipe with greps

  	```
  	cat SomeFile.txt | python egrep.py "[0-9]" | python line_count.py
  	```

  ```python
  # egrep.py
  import sys, re
  
  # sys.argv is the list of command-line arguments
  # sys.argv[0] is the name of the program itself
  # sys.argv[1] will be the regex specified at the command line
  regex = sys.argv[1]
  
  # for every line passed into the script
  for line in sys.stdin:
      # if it matches the regex, write it to stdout
      if re.search(regex, line):
          sys.stdout.write(line)
  ```

  ```python
      # line_cout.py
      import sys
      
      count = 0
      for line in sys.stdin:
          count += 1
      
      # print goes to sys.stdout
      print(count)
  ```

- Ex 2: Most common word of files
  ```terminal
  cat the_bible.txt `|` python most_common_words.py 10
  ```

```python
	# most_common_words.py
	import sys
	from collections import Counter
	
	# pass in number of words as first argument
	try:
	    num_words = int(sys.argv[1])
	except:
	    print("usage: most_common_words.py num_words")
	    sys.exit(1)   # nonzero exit code indicates error
	
	counter = Counter(word.lower()                      # lowercase words
	                  for line in sys.stdin
	                  for word in line.strip().split()  # split on spaces
	                  if word)                          # skip empty 'words'
	
	for word, count in counter.most_common(num_words):
	    sys.stdout.write(str(count))
	    sys.stdout.write("\t")
	    sys.stdout.write(word)
	    sys.stdout.write("\n")
	```


### 9.2 Reading Files
- Basic: Read all the domains of a file
	```python
	def get_domain(email_address: str) -> str:
	    """Split on '@' and return the last piece"""
	    return email_address.lower().split("@")[-1]
	
	# a couple of tests
	assert get_domain('joelgrus@gmail.com') == 'gmail.com'
	assert get_domain('joel@m.datasciencester.com') == 'm.datasciencester.com'
	
	from collections import Counter
	
	with open('email_addresses.txt', 'r') as f:
	    domain_counts = Counter(get_domain(line.strip())
	                            for line in f
	                            if "@" in line)
	```


### 9.3 Scraping the Web
- Tools
	- Use Beautiful Soup Library to build a tree out of the web page
	- Use Request library to make HTTP request
	- Parser:  html5lib
	- python -m pip install beautifulsoup4 requests html5lib

- Ex: Scrape house of representative _[web](https://www.house.gov/representatives)_ who mentioned data in their press link
	_HTML: view source_
	```html
	<td>
	  <a href="https://jayapal.house.gov">Jayapal, Pramila</a>
	</td>
	```

	_Code 1: Find the Valid URLS of representative_
	```python
	from bs4 import BeautifulSoup
	import requests
	
	url = "https://www.house.gov/representatives"
	text = requests.get(url).text
	soup = BeautifulSoup(text, "html5lib")

	# Must start with http:// or https://
	# Must end with .house.gov or .house.gov/
	regex = r"^https?://.*\.house\.gov/?$"

	# Let's write some tests!
	assert re.match(regex, "http://joel.house.gov")
	assert re.match(regex, "https://joel.house.gov")

	all_urls = [a['href']
	            for a in soup('a')
	            if a.has_attr('href')]
	
	# And now apply
	good_urls = [url for url in all_urls if re.match(regex, url)]

	print(len(set(all_urls)))  # 965 for me, way too many
```

_Code 2: Find the Press Link_
-  Test code: each press link is a relative link, which means we need to remember the originating site
   ```python
   html = requests.get('https://jayapal.house.gov').text
   soup = BeautifulSoup(html, 'html5lib')
   
   # Use a set because the links might appear multiple times.
   links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
   
   print(links) # {'/media/press-releases'}
   ```

- Actual Code: dict of representative to links
```python
	from typing import Dict, Set
	
	press_releases: Dict[str, Set[str]] = {}
	
	for house_url in good_urls:
	    html = requests.get(house_url).text
	    soup = BeautifulSoup(html, 'html5lib')
	    pr_links = {a['href'] for a in soup('a') if 'press releases'
	                                             in a.text.lower()}
	    print(f"{house_url}: {pr_links}")
	    press_releases[house_url] = pr_links
```

_Code 3: Did press link mention data?_
```python
for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        text = requests.get(url).text

        if paragraph_mentions(text, 'data'):
            print(f"{house_url}")
            break  # done with this house_url
```


### 9.4 Using the APIs

### 9.5 Using Twitter APIs
- Logic
    1. Go to https://developer.twitter.com/.
    2. If you are not signed in, click “Sign in” and enter your Twitter username and password.
    3. Click Apply to apply for a developer account.
    4. Request access for your own personal use.
    5. Fill out the application. It requires 300 words (really) on why you need access, so to get over the limit you could tell them about this book and how much you’re enjoying it.
    6. Wait some indefinite amount of time.
    7. If you know someone who works at Twitter, email them and ask them if they can expedite your application. Otherwise, keep waiting.
    8. Once you get approved, go back to developer.twitter.com, find the “Apps” section, and click “Create an app.”
    9. Fill out all the required fields (again, if you need extra characters for the description, you could talk about this book and how edifying you’re finding it).
    10. Click CREATE.
- _Code1 _: Create twython Instance

  ```python
  import os

  # Feel free to plug your key and secret in directly
  CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
  CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")

  import webbrowser
  from twython import Twython
  
  # Get a temporary client to retrieve an authentication URL
  temp_client = Twython(CONSUMER_KEY, CONSUMER_SECRET)
  temp_creds = temp_client.get_authentication_tokens()
  url = temp_creds['auth_url']
  
  # Now visit that URL to authorize the application and get a PIN
  print(f"go visit {url} and get the PIN code and paste it below")
  webbrowser.open(url)
  PIN_CODE = input("please enter the PIN code: ")
  
  # Now we use that PIN_CODE to get the actual tokens
  auth_client = Twython(CONSUMER_KEY,
                        CONSUMER_SECRET,
                        temp_creds['oauth_token'],
                        temp_creds['oauth_token_secret'])
  final_step = auth_client.get_authorized_tokens(PIN_CODE)
  ACCESS_TOKEN = final_step['oauth_token']
  ACCESS_TOKEN_SECRET = final_step['oauth_token_secret']
  
  # And get a new Twython instance using them.
  twitter = Twython(CONSUMER_KEY,
                    CONSUMER_SECRET,
                    ACCESS_TOKEN,
                    ACCESS_TOKEN_SECRET)
  ```

_Code2a _: Search whether tweet contains the word data
```python
# Search for tweets containing the phrase "data science"
for status in twitter.search(q='"data science"')["statuses"]:
    user = status["user"]["screen_name"]
    text = status["text"]
    print(f"{user}: {text}\n")
```

_Code2b: Using Streaming API to collect all data continuously_
- define on_success and on_error
```python
from twython import TwythonStreamer

# Appending data to a global variable is pretty poor form
# but it makes the example much simpler
tweets = []

class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        """
        What do we do when Twitter sends us data?
        Here data will be a Python dict representing a tweet.
        """
        # We only want to collect English-language tweets
        if data.get('lang') == 'en':
            tweets.append(data)
            print(f"received tweet #{len(tweets)}")

        # Stop when we've collected enough
        if len(tweets) >= 100:
            self.disconnect()

    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()

stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
                    ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# starts consuming public statuses that contain the keyword 'data'
stream.statuses.filter(track='data')

# if instead we wanted to start consuming a sample of *all* public statuses
# stream.statuses.sample()


top_hashtags = Counter(hashtag['text'].lower()
                       for tweet in tweets
                       for hashtag in tweet["entities"]["hashtags"])

print(top_hashtags.most_common(5))
```


# 10 Working With Data
### 10.1 Exploring Your data

_One Dimension: just one axis_
- histogram  a distribution
```python
	from typing import List, Dict
	from collections import Counter
	import math
	
	import matplotlib.pyplot as plt
	
	def bucketize(point: float, bucket_size: float) -> float:
	    """Floor the point to the next lower multiple of bucket_size"""
	    return bucket_size * math.floor(point / bucket_size)
	
	def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
	    """Buckets the points and counts how many in each bucket"""
	    return Counter(bucketize(point, bucket_size) for point in points)
	
	def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
	    histogram = make_histogram(points, bucket_size)
	    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
	    plt.title(title)


# normal distribution with mean 0, standard deviation 57
import random
# uniform between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10000)]
plot_histogram(uniform, 10, "Uniform Histogram")

```

_Two Dimensions: Compare 2 Variables Via Scatter Plot_
```python

# Define 2 axis input
def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray',  label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()


```

_Many Dimensions_
- use correlation matrix
```python
from scratch.linear_algebra import Matrix, Vector, make_matrix

def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Returns the len(data) x len(data) matrix whose (i, j)-th entry
    is the correlation between data[i] and data[j]
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len(data), correlation_ij)


# corr_data is a list of four 100-d vectors
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):

        # Scatter column_j on the x-axis vs. column_i on the y-axis
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

        # unless i == j, in which case show the series name
        else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                xycoords='axes fraction',
                                ha="center", va="center")

        # Then hide axis labels except left and bottom charts
        if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

# Fix the bottom-right and top-left axis labels, which are wrong because
# their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())

plt.show()


```


### 10.2 Using Named Tuples To Model Data
- Rationale:
    - using dict is error prone because of undefined keys.
    - NamedTuple is a tuple, and is better than ordinary tuple
        - because we can access with named slots
        - like tuple, it is also immutable

  ```python
  from collections import namedtuple

  StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
  price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)
  
  assert price.symbol == 'MSFT'
  assert price.closing_price == 106.03
  ```


### 10.3 Dataclasses
- Mutable data structure
- Dataclass usage may feel like a namedtuple but it is
    - data classes
        - are implemented as classes
        - is mutable
    - while nametuple is implemented as tuples (which is more compact)

- Implement with decorator
```python
	from dataclasses import dataclass
	
	@dataclass
	class StockPrice2:
	    symbol: str
	    date: datetime.date
	    closing_price: float
	
	    def is_high_tech(self) -> bool:
	        """It's a class, so we can add methods too"""
	        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
	
	price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)
	
	assert price2.symbol == 'MSFT'
	assert price2.closing_price == 106.03
	assert price2.is_high_tech()
```
### 10.4 Cleaning and Munging
- What to do with bad data (invalid format, etc..) ?
    1. Reject data
        1. maybe ok if we have billions of record
    2. Fix the source
    3. Do nothing and hope (haha..)

### 10.5 Manipulating data
- What columns, precision do we want to collect?
- This is where the business and domain logic first starts to creep in




### 10.6 Rescaling
- Refer to ==Book: Feature Scaling For Machine Learning== for more details

### 10.7 TQDM
```python
import tqdm

for i in tqdm.tqdm(range(100)):
    # do something slow
    _ = [random.random() for _ in range(1000000)]
```

### 10.8 Dimension Reduction





# 11 Machine Learning
### 11.1 Modeling
- Models are mathematical (or probabilistic) relationship that exists between different variables.


### 11.2 What is ML?
- machine learning to refer to creating and using models that are learned from data

### 11.3 Overfitting and Underfitting
- Refer to [[Book - Machine Learning System Design Interview - Detail]]  4.1.3 Regularization

### 11.4 Bias Variance TradeOffs

### 11.5 Feature Extraction and Selection
- IMOW: book is a bit shallow on this
    - Dimensional reduction
    - Regularizationn chang
    -



# 12 K Nearest Neighbor
### 12.1 The Model
- How it works
  1. Look at the K nearest points, as measured by a distance function
  2. The predicted label is the function of the the neighbor's label
  - But what about ties?
  - Pick one of the winners at random.
  - Weight the votes by distance and pick the weighted winner.
  - Reduce k until we find a unique winner.
  ```python
      def majority_vote(labels: List[str]) -> str:
          """Assumes that labels are ordered from nearest to farthest."""
          vote_counts = Counter(labels)
          winner, winner_count = vote_counts.most_common(1)[0]
          num_winners = len([count
                             for count in vote_counts.values()
                             if count == winner_count])
      
          if num_winners == 1:
              return winner
          else:
              # try again without the farthest
              return majority_vote(labels[:-1]) 
      
      # Tie, so look at first 4, then 'b'
      assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'
      ```

- Algorithm
  ```python
  from typing import NamedTuple
  from scratch.linear_algebra import Vector, distance
  
  class LabeledPoint(NamedTuple):
      point: Vector
      label: str
  
  def dot(v: Vector, w: Vector) -> float:
      return sum(v_i * w_i for v_i, w_i in zip(v, v))

  # sum of sqaures of 1 vector can be defined in terms of dot function
  def sum_of_squares(v: Vector) -> float:
      return dot(v, v)
      
  def squared_distance(v: Vector, w: Vector) -> float:
      """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
      return sum_of_squares(subtract(v, w))

  # Euclidean distance
  def distance(v: Vector, w: Vector) -> float:
      """Computes the distance between v and w"""
      return math.sqrt(squared_distance(v, w))
      
  def knn_classify(k: int,
                   labeled_points: List[LabeledPoint],
                   new_point: Vector) -> str:
  
      # Order the labeled points from nearest to farthest.
      by_distance = sorted(labeled_points,
                           key=lambda lp: distance(lp.point, new_point))
  
      # Find the labels for the k closest
      k_nearest_labels = [lp.label for lp in by_distance[:k]]
  
      # and let them vote.
      return majority_vote(k_nearest_labels)
  ```

- Vs KMeans Clustering
    - KNN is a supervised machine learning algorithm used for ==classification==
    - KMeans is an unsupervised machine learning algorithm used for clustering.


### 12.2 Example: Iris Dataset
```python
from typing import Tuple

# track how many times we see (predicted, actual)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)
```

### 12.3 Curse of Dimensionality
- Limitation of K nearest neighbor is high dimensions space are vast, increasing the chance the points are not close to each other.
    - two points are close only if they’re close in every dimension, and every extra dimension—even if just noise—is another opportunity for each point to be farther away from every other point. When you have a lot of dimensions, it’s likely that the closest points aren’t much closer than average, so two points being close doesn’t mean very much

### 12.4 Further Exploration



# 13 Naives Bayes
### 13.1 Really Dumb Spam Filter
- Given
    - S = Spam
    - X is a vector of word length n
- The probability of spam for the sentence is
  $$ P(S | X) = [P(X | S)\times P(S)] / P(X)  $$

### 13.2 More Sophisticated Spam Filter

- Need to address case  Numerical underflow
  $$P(X_1 = x_1,..X_n=x_n) = P(X_1=x_1|S) \times ... \times P(X_n=x_n | S)$$
  $$ = \exp[ log(P(X_1=x_1|S)) + log(P(X_n=x_n|S)) ]$$
  $$P(X_1=x_1|S) + .. + .. P(X_n=x_n|S)  $$

-  Also need to address : For rare words in spam text messages--> aka smoothing
   $$P(X_i | Spam) = (k + numSpamsContainingX_i)/(2k + numSpams )$$

### 13.3 Implementation
_Code: Naives Bayes_
```python
from typing import Set
import re
def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)  


from typing import NamedTuple
class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict
class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

	def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

   def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Otherwise add the log probability of _not_ seeing it,
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)
```

### 13.4 Testing
```python
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Should be about 0.83
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)
```
### 13.5 Using The Model
```python
from io import BytesIO  # So we can treat bytes as a file.
import requests         # To download the files, which
import tarfile          # are in .tar.bz format.

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

# This is where the data will end up,
# in /spam, /easy_ham, and /hard_ham subdirectories.
# Change this to where you want the data.
OUTPUT_DIR = 'spam_data'

for filename in FILES:
    # Use requests to get the file contents at each URL.
    content = requests.get(f"{BASE_URL}/{filename}").content

    # Wrap the in-memory bytes so we can use them as a "file."
    fin = BytesIO(content)

    # And extract all the files to the specified output dir.
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)


import glob, re
# modify the path to wherever you've put the files
path = 'spam_data/*/*'
data: List[Message] = []
# glob.glob returns every filename that matches the wildcarded path
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    # There are some garbage characters in the emails; the errors='ignore'
    # skips them instead of raising an exception.
    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break  # done with this file


import random
from scratch.machine_learning import split_data
random.seed(0)      # just so you get the same answers as me
train_messages, test_messages = split_data(data, 0.75)
model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter
predictions = [(message, model.predict(message.text))
               for message in test_messages]
# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)
print(confusion_matrix)

```

# 14 Simple Linear Regression
### 14.1 The Model
- Task: Predict time of use as a function of numFriends
  $$y_i = \beta x_i + \alpha  $$
    - such that
        - $y_i$ is the amount of time user i spend on platform
        - $x_i$ is the number of friends user_i has on the platform

- Code
```python
def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i


from scratch.linear_algebra import Vector
def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))



from typing import Tuple
from scratch.linear_algebra import Vector
from scratch.statistics import correlation, standard_deviation, mean
def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
```

- Goodness of fit: How well does our parameter fit the data?
    - R-squared (aka coefficient of determination) measures the fraction of total variation in the dependent variable (ie y) is captured by the variable
    -


	```python
	from scratch.statistics import de_mean
	
	def total_sum_of_squares(y: Vector) -> float:
	    """the total squared variation of y_i's from their mean"""
	    return sum(v ** 2 for v in de_mean(y))
	
	def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
	    """
	    the fraction of variation in y captured by the model, which equals
	    1 - the fraction of variation in y not captured by the model
	    """
	    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
	                  total_sum_of_squares(y))
	```

### 14.2 Using Gradient Descent
- In 14.1, we solved the close form of the linear equation where we solved the parameters $\alpha \beta$
- In this section, we use gradient descent as a way to approximate the solution
- theta = $\alpha , \beta$

  ```python
  import random
  import tqdm
  from scratch.gradient_descent import gradient_step
  
  num_epochs = 10000
  random.seed(0)
  
  guess = [random.random(), random.random()]  # choose random value to start
  
  learning_rate = 0.00001
  
  with tqdm.trange(num_epochs) as t:
      for _ in t:
          alpha, beta = guess
  
          # Partial derivative of loss with respect to alpha
          grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                       for x_i, y_i in zip(num_friends_good,
                                           daily_minutes_good))
  
          # Partial derivative of loss with respect to beta
          grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                       for x_i, y_i in zip(num_friends_good,
                                           daily_minutes_good))
  
          # Compute loss to stick in the tqdm description
          loss = sum_of_sqerrors(alpha, beta,
                                 num_friends_good, daily_minutes_good)
          t.set_description(f"loss: {loss:.3f}")
  
          # Finally, update the guess
          guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
  ```


### 14.3 Maximum Liklihood Estmation (MLE)
- In this section, we use MLE to explain why we use least squares fit to derive the closed form for parameters $\alpha, \beta$
- MLE estimates the parameters $\theta$ that makes the observed data most probable.
    - Minimizing the sum of squared errors is equivalent to maximizing the likelihood of the observed data.


# 15 Multiple Regression
### 15. 1 The Model
- In Chapter 14, we solve a linear equation with 2 parameters. In this chapter, we increase the number of parameters we will learn.

- Let's increase the num parameter we learn (go beyond a linear line)
  $$y_i = \alpha + \beta_1x_{i1} + ... + \beta_kx_{ik}$$
    - where
        - i = user i
        - k = number of parameters we are fitting

- model learn parameter as a vector 	$$\beta = [\alpha, \beta_1, .., \beta_k]$$
- predict code
  ```python
  def predict(x: Vector, beta: Vector) -> float:
      """assumes that the first element of x is 1"""
      return dot(x, beta)
  ```


### 15. 2 Further Assumptions of Least Squares Model
- Key assumptions
    - All the features are linearly independent
    - The parameter of input vector x are uncorrelated with the error


### 15.3 Fitting the Model
_Code: Helper Functions_
```python
	from typing import List
	
	def error(x: Vector, y: float, beta: Vector) -> float:
	    return predict(x, beta) - y
	
	def squared_error(x: Vector, y: float, beta: Vector) -> float:
	    return error(x, y, beta) ** 2
	
	def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
	    err = error(x, y, beta)
	    return [2 * err * x_i for x_i in x]
    
```


_Code: Least Squares Fit_
```python
	import random
	import tqdm
	from scratch.linear_algebra import vector_mean
	from scratch.gradient_descent import gradient_step
	
	def least_squares_fit(xs: List[Vector],
	                      ys: List[float],
	                      learning_rate: float = 0.001,
	                      num_steps: int = 1000,
	                      batch_size: int = 1) -> Vector:
	    """
	    Find the beta that minimizes the sum of squared errors
	    assuming the model y = dot(x, beta).
	    """
	    # Start with a random guess
	    guess = [random.random() for _ in xs[0]]
	
	    for _ in tqdm.trange(num_steps, desc="least squares fit"):
	        for start in range(0, len(xs), batch_size):
	            batch_xs = xs[start:start+batch_size]
	            batch_ys = ys[start:start+batch_size]
	
	            gradient = vector_mean([sqerror_gradient(x, y, guess)
	                                    for x, y in zip(batch_xs, batch_ys)])
	            guess = gradient_step(guess, gradient, -learning_rate)
	
	    return guess
```


### 15.4 Interpreting the Model
- The value of the learned coefficients represents the importance of that features, assuming ==all else being equal (big caveat)==
    - Ex: all else being equal, having one friend translate to 2 minutes on the app
    - Parameters do not explain how the variables interact with each other.


### 15.5 Goodness of Fit
```python
	from scratch.simple_linear_regression import total_sum_of_squares
	
	def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
	    sum_of_squared_errors = sum(error(x, y, beta) ** 2
	                                for x, y in zip(xs, ys))
	    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)
```


### 15.6 The Bootstrap
-   Bootstrapping is a method of inferring results for a population from results found on a collection of ==smaller random ==samples== of that population, **using replacement during the sampling process**.

- Code: utility function
```python
	from typing import TypeVar, Callable
	
	X = TypeVar('X')        # Generic type for data
	Stat = TypeVar('Stat')  # Generic type for "statistic"
	
	def bootstrap_sample(data: List[X]) -> List[X]:
	    """randomly samples len(data) elements with replacement"""
	    return [random.choice(data) for _ in data]
	
	def bootstrap_statistic(data: List[X],
	                        stats_fn: Callable[[List[X]], Stat],
	                        num_samples: int) -> List[Stat]:
	    """evaluates stats_fn on num_samples bootstrap samples from data"""
	    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


```

- Code: usage
```python
# 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]
medians_close = bootstrap_statistic(close_to_100, median, 100)


# 101 points, 50 of them near 0, 50 of them near 200
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])
medians_far = bootstrap_statistic(far_from_100, median, 100)
```


### 15.7 Standard Errors of Regression Coefficient


### 15.8 Regularization
- Regularization is an approach in which we add to the error term a penalty that gets larger as beta gets larger. We then minimize the combined error and penalty. The more importance we place on the penalty term, the more we discourage large coefficients.
- 2 Types in LInear Regression
    1. Ridge (aka L2)
        1. shrink the coefficient overall
    2. Lasso (aka L1)
        1. drive coefficient to 0 to learn more sparse models
- Code: Ridge (L2)
```python
	# alpha is a *hyperparameter* controlling how harsh the penalty is.
	# Sometimes it's called "lambda" but that already means something in Python.
	def ridge_penalty(beta: Vector, alpha: float) -> float:
	    return alpha * dot(beta[1:], beta[1:])
	
	def squared_error_ridge(x: Vector,
	                        y: float,
	                        beta: Vector,
	                        alpha: float) -> float:
	    """estimate error plus ridge penalty on beta"""
	    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


	from scratch.linear_algebra import add
	def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
	    """gradient of just the ridge penalty"""
	    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]
	
	def sqerror_ridge_gradient(x: Vector,
	                           y: float,
	                           beta: Vector,
	                           alpha: float) -> Vector:
	    """
	    the gradient corresponding to the ith squared error term
	    including the ridge penalty
	    """
	    return add(sqerror_gradient(x, y, beta),
	               ridge_penalty_gradient(beta, alpha))	
```

```python
	# TEST CODE
	random.seed(0)
	beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0,  # alpha
	                                 learning_rate, 5000, 25)
```

- Code: Lasso
```python
	def lasso_penalty(beta, alpha):
	    return alpha * sum(abs(beta_i) for beta_i in beta[1:])
```

# 16 Logistic Regression
### 16.1 The Problem
- Potential issues with linear regression models
    - Output model range does not match our end use case of 0 or 1
    - Linear regression assumes
        1. Linear relationship between X and the mean of Y
            1. Not true if R coefficient is positive
        2. Variance of residual is the same for any value of X
        3. Observations are independent of each other

- Logistic function
  $$y = \frac{1}{1 + e^{-x}}$$
  $$x = f(x_i\beta)$$


### 16.2 The Logistic Function


### 16.3 Applying the Model


### 16.4 Goodness of Fit

### 16.5 Support Vector Machines



# 17 Decision Tree
### 17.1 What is a Decision Tree
- A decision tree uses a tree structure to represent a number of possible decision paths and an outcome for each path.

### 17.2 Entropy
- Entropy measure how much information/randomness there is
    - Applied to ML,
        - high entropy means there are multiple classes label in the set
        - low entropy means there are few (or 1) class label in the set

- Mathematics
  $$H(S) = -p_1\log_2p_1 - ... -p_n\log_2p_n $$
  - S = Partition Set
  - $p_n$ is the proportion of class | label in the set S
  - ==Entropy is lowest when the $p_n$ is 0 or 1 ==

- Code
```python
	from typing import List, Any
	import math
	from collections import Counter
	
	def entropy(class_probabilities: List[float]) -> float:
	    """Given a list of class probabilities, compute the entropy"""
	    return sum(-p * math.log(p, 2)
	               for p in class_probabilities
	               if p > 0)                     # ignore zero probabilities
	
	def class_probabilities(labels: List[Any]) -> List[float]:
	    total_count = len(labels)
	    return [count / total_count
	            for count in Counter(labels).values()]
	
	def data_entropy(labels: List[Any]) -> float:
	    return entropy(class_probabilities(labels))

```




### 17.3 Entropy of a Partition
- Lowest entropy of a  partition is when all the member in a set belongs to a label class

- Mathematics to calculate the entropy of a partition
    - ==If we partition the data by a feature (ie gender) and its value(ie female), we will create multiple sets, each set will have data with labels from class 1 to m==
      $$H_{partition} = q_1H(S_1) + ... + q_mH(S_m)$$
      - where
      - $q_m$ is the proportion of data with class label m
      - $S_m$ is the resulting set where all the data has gender=male (as an example), but the remaining features (age, race) will each be split until we reach the leaf node.
- Code
  ```python
  def partition_entropy(subsets: List[List[Any]]) -> float:
      """Returns the entropy from this partition of data into subsets"""
      # subsets is S1, S2, .., S_m in our equation above 
      total_count = sum(len(subset) for subset in subsets)
  
      return sum(data_entropy(subset) * len(subset) / total_count
                 for subset in subsets)
  ```


### 17.4 Creating a Decision Tree
_Algorithm-Simplified_
1. If the data all have the same label, create a leaf node that predicts that label and then stop.
2. If the list of attributes is empty (i.e., there are no more possible questions to ask), create a leaf node that predicts the most common label and then stop.
3. Otherwise, try partitioning the data by each of the attributes (aka feature).
4. Choose the partition with the ==lowest partition entropy==.
5. Add a decision node based on the chosen attribute.
6. Recur on each partitioned subset using the remaining attributes.

##### 17.4.1 Example: Decision tree to see whether to interview someone based on the features level, language, tweets, and phd

_Mock Data_
```python
from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data

                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False)]
```

_Partition The Set of Candidate Based on a attribute/feature_
```python
from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T')  # generic type for inputs

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)    # add input to the correct partition
    return partitions
```


```python
def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)
```

_Test Code_
```python
for key in ['level','lang','tweets','phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))

assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well')  < 0.70
assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well')   < 0.87
assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well')    < 0.90
```


### 17.5 Putting It All Together
_Code: Inference Logic_
```python
from typing import NamedTuple, Union, Any

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]


# This is the result of our training the tree
hiring_tree = Split('level', {   # first, consider "level"
    'Junior': Split('phd', {     # if level is "Junior", next look at "phd"
        False: Leaf(True),       #   if "phd" is False, predict True
        True: Leaf(False)        #   if "phd" is True, predict False
    }),
    'Mid': Leaf(True),           # if level is "Mid", just predict True
    'Senior': Split('tweets', {  # if level is "Senior", look at "tweets"
        False: Leaf(False),      #   if "tweets" is False, predict False
        True: Leaf(True)         #   if "tweets" is True, predict True
    })
})
```


```python
def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value

    # Otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values are subtrees to consider next
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:   # If no subtree for key,
        return tree.default_value          # return the default value.

    subtree = tree.subtrees[subtree_key]   # Choose the appropriate subtree
    return classify(subtree, input)        # and use it to classify the input.
```


_Code: Build The Tree_
```python
def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise split by the best attribute

    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # Recursively build the subtrees
    subtrees = {attribute_value : build_tree_id3(subset,
                                                 new_attributes,
                                                 target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)
```


### 17.6 Random Forests
- Reduce decision tree's tendency to overfit by bootsgtapping, where
    - bootstrap aggregation (aka bagging): each tree is selected with different sampled subset of data
    - randomly select subset of attributes




# 18 Neural Network
### 18.1 Perceptrons


### 18.2 Feed Forward Neural Network


### 18.3 Back propagation


### 18.4 Example: Fiz Fuzz



# 19 Deep Learning
### 19.1 The Tensor

### 19.2 The Layer Abstraction


### 19.3 Linear Layer

### 19.4 NN As a Sequence of Layers


### 19.5 Loss and Optimization


### 19.6 Example: XOR Revisited

### 19.7 Other Activation Functions


### 19.8 FizzBuzz Revisited


### 19.9 Softmaxes and Cross Entropy

### 19.10 Dropout


### 19.11 Example: MNIST


### 19.12 Saving and Loading Models



# 20 Clustering
### 20.1 The Idea (K-Means)
- Partition the inputs into sets $S_1, .., S_k$  in a way to minimize the squared distance of each point from the $cluster_k$ mean.
    - This is a top down approach, starting from the mean of k clusters. Compared to the bottom up hierarchical clustering in section 20.6

### 20.2 The Model
- Algorithm
    1. Start with a set of k-means, which are points in d-dimensional space.
    2. Assign each point to the mean to which it is closest.
    3. If no point’s assignment has changed, stop and keep the clusters.
    4. If some point’s assignment has changed, recompute the means and return to step 2.

_Code: How many coordinates 2 vectors differ in_
```python
	from scratch.linear_algebra import Vector
	
	def num_differences(v1: Vector, v2: Vector) -> int:
	    assert len(v1) == len(v2)
	    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])
	
	assert num_differences([1, 2, 3], [2, 1, 3]) == 2
	assert num_differences([1, 2], [1, 2]) == 0
```

_Code: Given some vectors and their assignments to clusters, compute the mean of the clusters_
```python
	from typing import List
	from scratch.linear_algebra import vector_mean
	
	def cluster_means(k: int,
	                  inputs: List[Vector],
	                  assignments: List[int]) -> List[Vector]:
	    # clusters[i] contains the inputs whose assignment is i
	    clusters = [[] for i in range(k)]
	    for input, assignment in zip(inputs, assignments):
	        clusters[assignment].append(input)
	
	    # if a cluster is empty, just use a random point
	    return [vector_mean(cluster) if cluster else random.choice(inputs)
	            for cluster in clusters]
```

_Code: End to End_
```python
import itertools
import random
import tqdm
from scratch.linear_algebra import squared_distance

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k                      # number of clusters
        self.means = None

    def classify(self, input: Vector) -> int:
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")
```


### 20.3 Example: Meetups


### 20.4 Choosing K
- various ways to choose k
    - Plot squared distance error as a function of k; choose where the graph bends

_Code_
```python
	from matplotlib import pyplot as plt
	
	def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
	    """finds the total squared error from k-means clustering the inputs"""
	    clusterer = KMeans(k)
	    clusterer.train(inputs)
	    means = clusterer.means
	    assignments = [clusterer.classify(input) for input in inputs]
	
	    return sum(squared_distance(input, means[cluster])
	               for input, cluster in zip(inputs, assignments))


	# now plot from 1 up to len(inputs) clusters
	ks = range(1, len(inputs) + 1)
	errors = [squared_clustering_errors(inputs, k) for k in ks]
	
	plt.plot(ks, errors)
	plt.xticks(ks)
	plt.xlabel("k")
	plt.ylabel("total squared error")
	plt.title("Total Error vs. # of Clusters")
	plt.show()
```

### 20.5 Example: Clustering Colors


### 20.6 Bottom Up Hierarchical Clustering
- Algorithm
    1. Make each input its own cluster of one.
    2. As long as there are multiple clusters remaining, find the two closest clusters and merge them.


_Code: Modeling_
```python
	from typing import NamedTuple, Union
	
	class Leaf(NamedTuple):
	    value: Vector

	class Merged(NamedTuple):
	    children: tuple
	    order: int # the sequence of the merge

	leaf1 = Leaf([10,  20])
	leaf2 = Leaf([30, -15])
	merged = Merged((leaf1, leaf2), order=1)
	Cluster = Union[Leaf, Merged]
```


_Code: Helper Function_
```python
	# Return all values contained in cluster recursively
	def get_values(cluster: Cluster) -> List[Vector]:
	    if isinstance(cluster, Leaf):
	        return [cluster.value]
	    else:
	        return [value
	                for child in cluster.children
	                for value in get_values(child)]
	assert get_values(merged) == [[10, 20], [30, -15]]


	# Metric for distance
	from typing import Callable
	from scratch.linear_algebra import distance
	def cluster_distance(cluster1: Cluster,
	                     cluster2: Cluster,
	                     distance_agg: Callable = min) -> float:
	    """
	    compute all the pairwise distances between cluster1 and cluster2
	    and apply the aggregation function _distance_agg_ to the resulting list
	    """
	    return distance_agg([distance(v1, v2)
	                         for v1 in get_values(cluster1)
	                         for v2 in get_values(cluster2)])


	# Pts will be merged from bottom up; keep track of the order
	def get_merge_order(cluster: Cluster) -> float:
	    if isinstance(cluster, Leaf):
	        return float('inf')  # was never merged
	    else:
	        return cluster.order


	def get_children(cluster: Cluster):
	    if isinstance(cluster, Leaf):
	        raise TypeError("Leaf has no children")
	    else:
	        return cluster.children
```

_Code: Bottom Up_
```python
	def bottom_up_cluster(inputs: List[Vector],
	                      distance_agg: Callable = min) -> Cluster:
	    # Start with all leaves
	    clusters: List[Cluster] = [Leaf(input) for input in inputs]
	
	    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
	        return cluster_distance(pair[0], pair[1], distance_agg)
	
	    # as long as we have more than one cluster left...
	    while len(clusters) > 1:
	        # find the two closest clusters
	        c1, c2 = min(((cluster1, cluster2)
	                      for i, cluster1 in enumerate(clusters)
	                      for cluster2 in clusters[:i]),
	                      key=pair_distance)
	
	        # remove them from the list of clusters
	        clusters = [c for c in clusters if c != c1 and c != c2]
	
	        # merge them, using merge_order = # of clusters left
	        merged_cluster = Merged((c1, c2), order=len(clusters))
	
	        # and add their merge
	        clusters.append(merged_cluster)
	
	    # when there's only one cluster left, return it
	    return clusters[0]
```


# 21 Natural Language Processing
### 21.1 Word Clouds
_Given (Word, jobListing cnt, resume cnt)_
```python
	data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]
```

_Code to Plot Word Cloud_
```python
	from matplotlib import pyplot as plt
	
	def text_size(total: int) -> float:
	    """equals 8 if total is 0, 28 if total is 200"""
	    return 8 + total / 200 * 20
	
	for word, job_popularity, resume_popularity in data:
	    plt.text(job_popularity, resume_popularity, word,
	             ha='center', va='center',
	             size=text_size(job_popularity + resume_popularity))
	plt.xlabel("Popularity on Job Postings")
	plt.ylabel("Popularity on Resumes")
	plt.axis([0, 100, 0, 100])
	plt.xticks([])
	plt.yticks([])
	plt.show()
```

### 21.2 n-Gram Language Models


### 21.3 Grammers

### 21.4 Gibbs Sampling


### 21.5 ==Topics Modeling==


### 21.6 ==Word Vectors==
_Background_
- Word vectors have the nice property that similar words appear near each other


_Code: Similarity Measurement Via Cosine Similarity_
```python
	from scratch.linear_algebra import dot, Vector
	import math
	
	def cosine_similarity(v1: Vector, v2: Vector) -> float:
	    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))
```

_Code: Create Training Sentences_
```python
	colors = ["red", "green", "blue", "yellow", "black", ""]
	nouns = ["bed", "car", "boat", "cat"]
	verbs = ["is", "was", "seems"]
	adverbs = ["very", "quite", "extremely", ""]
	adjectives = ["slow", "fast", "soft", "hard"]
	
	def make_sentence() -> str:
	    return " ".join([
	        "The",
	        random.choice(colors),
	        random.choice(nouns),
	        random.choice(verbs),
	        random.choice(adverbs),
	        random.choice(adjectives),
	        "."
	    ])
	
	NUM_SENTENCES = 50
	
	random.seed(0)
	sentences = [make_sentence() for _ in range(NUM_SENTENCES)]
```

_Code: Vocabulary Converts Word to ID_
```python
	from scratch.deep_learning import Tensor
	
	class Vocabulary:
	    def __init__(self, words: List[str] = None) -> None:
	        self.w2i: Dict[str, int] = {}  # mapping word -> word_id
	        self.i2w: Dict[int, str] = {}  # mapping word_id -> word
	
	        for word in (words or []):     # If words were provided,
	            self.add(word)             # add them.
	
	    @property
	    def size(self) -> int:
	        """how many words are in the vocabulary"""
	        return len(self.w2i)
	
	    def add(self, word: str) -> None:
	        if word not in self.w2i:        # If the word is new to us:
	            word_id = len(self.w2i)     # Find the next id.
	            self.w2i[word] = word_id    # Add to the word -> word_id map.
	            self.i2w[word_id] = word    # Add to the word_id -> word map.
	
	    def get_id(self, word: str) -> int:
	        """return the id of the word (or None)"""
	        return self.w2i.get(word)
	
	    def get_word(self, word_id: int) -> str:
	        """return the word with the given id (or None)"""
	        return self.i2w.get(word_id)
	
	    def one_hot_encode(self, word: str) -> Tensor:
	        word_id = self.get_id(word)
	        assert word_id is not None, f"unknown word {word}"
	
	        return [1.0 if i == word_id else 0.0 for i in range(self.size)]
```

_Code: Save and Load Vobcabulary_
```python
	import json
	
	def save_vocab(vocab: Vocabulary, filename: str) -> None:
	    with open(filename, 'w') as f:
	        json.dump(vocab.w2i, f)       # Only need to save w2i
	
	def load_vocab(filename: str) -> Vocabulary:
	    vocab = Vocabulary()
	    with open(filename) as f:
	        # Load w2i and generate i2w from it
	        vocab.w2i = json.load(f)
	        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
	    return vocab
```

_Code: Train Embedding Via SkipGram_
- Concepts
    - Skip-gram takes as input a word and generates probabilities for what words are likely to be seen near it. We will feed it training pairs (word, nearby_word) and try to minimize the SoftmaxCrossEntropy loss
    - Embedding layer takes a word-id input and returns a vector.  We will implement it as a lookup table {word_id -> vector_i }
    -  word_id --> linear layer --> softmax
        - dimension of word_id and linear layer = size of vocabulary

_Code: Embedding _
```python
from typing import Iterable
from scratch.deep_learning import Layer, Tensor, random_tensor, zeros_like

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # One vector of size embedding_dim for each desired embedding
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)

        # Save last input id
        self.last_input_id = None

	def forward(self, input_id: int) -> Tensor:
        """Just select the embedding vector corresponding to the input id"""
        self.input_id = input_id    # remember for use in backpropagation

        return self.embeddings[input_id]

	def backward(self, gradient: Tensor) -> None:
        # Zero out the gradient corresponding to the last input.
        # This is way cheaper than creating a new all-zero tensor each time.
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row

        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient

	def params(self) -> Iterable[Tensor]:
        return [self.embeddings]

    def grads(self) -> Iterable[Tensor]:
        return [self.grad]
```

_Code: Text Embedding_
```python
class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        # Call the superclass constructor
        super().__init__(vocab.size, embedding_dim)

        # And hang onto the vocab
        self.vocab = vocab

	def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None

	def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        """Returns the n closest words based on cosine similarity"""
        vector = self[word]

        # Compute pairs (similarity, other_word), and sort most similar first
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)

        return scores[:n]
```

_Code: Create Training Data_
```python
	import re

	# This is not a great regex, but it works on our data.
	tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower())
	                       for sentence in sentences]

	# Create a vocabulary (that is, a mapping word -> word_id) based on our text.
	vocab = Vocabulary(word
	                   for sentence_words in tokenized_sentences
	                   for word in sentence_words)



	from scratch.deep_learning import Tensor, one_hot_encode
	inputs: List[int] = []
	targets: List[Tensor] = []
	
	for sentence in tokenized_sentences:
	    for i, word in enumerate(sentence):          # For each word
	        for j in [i - 2, i - 1, i + 1, i + 2]:   # take the nearby locations
	            if 0 <= j < len(sentence):           # that aren't out of bounds
	                nearby_word = sentence[j]        # and get those words.
	
	                # Add an input that's the original word_id
	                inputs.append(vocab.get_id(word))
	
	                # Add a target that's the one-hot-encoded nearby word
	                targets.append(vocab.one_hot_encode(nearby_word))
```

_Code: Create Model_
```python
	from scratch.deep_learning import Sequential, Linear
	
	random.seed(0)
	EMBEDDING_DIM = 5  # seems like a good size
	
	# Define the embedding layer separately, so we can reference it.
	embedding = TextEmbedding(vocab=vocab, embedding_dim=EMBEDDING_DIM)
	
	model = Sequential([
	    # Given a word (as a vector of word_ids), look up its embedding.
	    embedding,
	    # And use a linear layer to compute scores for "nearby words."
	    Linear(input_dim=EMBEDDING_DIM, output_dim=vocab.size)
	])
```

_Code: Training Loop_
```python
	from scratch.deep_learning import SoftmaxCrossEntropy, Momentum, GradientDescent
	
	loss = SoftmaxCrossEntropy()
	optimizer = GradientDescent(learning_rate=0.01)
	
	for epoch in range(100):
	    epoch_loss = 0.0
	    for input, target in zip(inputs, targets):
	        predicted = model.forward(input)
	        epoch_loss += loss.loss(predicted, target)
	        gradient = loss.gradient(predicted, target)
	        model.backward(gradient)
	        optimizer.step(model)
	    print(epoch, epoch_loss)            # Print the loss
	    print(embedding.closest("black"))   # and also a few nearest words
	    print(embedding.closest("slow"))    # so we can see what's being
	    print(embedding.closest("car"))     # learned.
```

_Code: Analysis_
```python
	pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
	         for w1 in vocab.w2i
	         for w2 in vocab.w2i
	         if w1 < w2]
	pairs.sort(reverse=True)


	# PLOT
	from scratch.working_with_data import pca, transform
	import matplotlib.pyplot as plt
	
	# Extract the first two principal components and transform the word vectors
	components = pca(embedding.embeddings, 2)
	transformed = transform(embedding.embeddings, components)
	
	# Scatter the points (and make them white so they're "invisible")
	fig, ax = plt.subplots()
	ax.scatter(*zip(*transformed), marker='.', color='w')
	
	# Add annotations for each word at its transformed location
	for word, idx in vocab.w2i.items():
	    ax.annotate(word, transformed[idx])
	
	# And hide the axes
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	plt.show()
```


### 21.7 RNN

### 21.8 Example: Using a Character Level RNN




# 22 Network Analysis (Graph)
### 22.1 Betweenness Centrality

### 22.2 Eigenvector Centrality

### 22.3 Directed Graphs and Page Ranks



# 23 Recommender System
### 23.1 Manual Curations
```python
	users_interests = [
	    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
	    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
	    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
	    ["R", "Python", "statistics", "regression", "probability"],
	    ["machine learning", "regression", "decision trees", "libsvm"],
	    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
	    ["statistics", "probability", "mathematics", "theory"],
	    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
	    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
	    ["Hadoop", "Java", "MapReduce", "Big Data"],
	    ["statistics", "R", "statsmodels"],
	    ["C++", "deep learning", "artificial intelligence", "probability"],
	    ["pandas", "R", "Python"],
	    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
	    ["libsvm", "regression", "support vector machines"]
	]
```


### 23.2 Recommending What's Popular
```python
		# Flatten nested list
	from collections import Counter
	popular_interests = Counter(interest
	                            for user_interests in users_interests
	                            for interest in user_interests)

	# Suggest the most popular interests the user currently does not have
	# CON: new users will get everything; not very personalized
	from typing import List, Tuple
	def most_popular_new_interests(
	        user_interests: List[str],
	        max_results: int = 5) -> List[Tuple[str, int]]:
	    suggestions = [(interest, frequency)
	                   for interest, frequency in popular_interests.most_common()
	                   if interest not in user_interests]
	    return suggestions[:max_results]

```


### 23.3 User Based Collaborative Filtering
- Concept
    - Recommend other similar users who has similar interests
    - Create a user vector where each position represent a interest
    - Cosine similarity to find close users

_CODE:_
```python

unique_interests = sorted({interest
                           for user_interests in users_interests
                           for interest in user_interests})


# -------------------------------------------------------
# Create user vector
# -------------------------------------------------------
def make_user_interest_vector(user_interests: List[str]) -> List[int]:
    """
    Given a list of interests, produce a vector whose ith element is 1
    if unique_interests[i] is in the list, 0 otherwise
    """
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

user_interest_vectors = [make_user_interest_vector(user_interests)
                         for user_interests in users_interests]


# -------------------------------------------------------
# Find other simlar users
# -------------------------------------------------------
from scratch.nlp import cosine_similarity
user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_vectors]
                     for interest_vector_i in user_interest_vectors]
def most_similar_users_to(user_id: int) -> List[Tuple[int, float]]:
    pairs = [(other_user_id, similarity)                      # Find other
             for other_user_id, similarity in                 # users with
                enumerate(user_similarities[user_id])         # nonzero
             if user_id != other_user_id and similarity > 0]  # similarity.

    return sorted(pairs,                                      # Sort them
                  key=lambda pair: pair[-1],                  # most similar
                  reverse=True)                               # first.


# -------------------------------------------------------
# Suggest other users: 
#    - for each S similar users
#    - add each S similar user's simlarity score to interest score dict
#    - return top k score interest
# -------------------------------------------------------
from collections import defaultdict
def user_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
    # Sum up the similarities
    suggestions: Dict[str, float] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # Convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],  # weight
                         reverse=True)

    # And (maybe) exclude already interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]
```


### 23.4 Item Based Collaborative Filtering
- Compute similarities between items (aka interest directly)
    - suggest interests similar to the user's current interest

_CODE_
```python
# Transpose our preivous user-to-interest vector to interest-to-user vector
interest_user_matrix = [[user_interest_vector[j]
                        for user_interest_vector in user_interest_vectors]
                        for j, _ in enumerate(unique_interests)]

# `unique_interests[0]` is Big Data, and so 
#`interest_user_matrix[0]` is: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]


def most_similar_interests_to(interest_id: int):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)

def item_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
    # Add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # Sort them by weight
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]
```


### 23.5 Matrix Factorization
- Theory
    - Users and items are represented has a latent type (ie embedding)
        - User matrix = numUsers by vector dim matrix
    - Model learn the representation such that similarity_score(embedding_user, embedding_item) ~= label(ie rating)

- Example - MoveLens Ratings:
    - [data](http://files.grouplens.org/datasets/movielens/ml-100k.zip).

_Code: Ingest Data_
```python
	# Class models
	from typing import NamedTuple
	class Rating(NamedTuple):
	    user_id: str
	    movie_id: str
	    rating: float
	
	
	# Import Movie ratings
	import csv
	# We specify this encoding to avoid a UnicodeDecodeError.
	# See: https://stackoverflow.com/a/53136168/1076346.
	with open(MOVIES, encoding="iso-8859-1") as f:
	    reader = csv.reader(f, delimiter="|")
	    movies = {movie_id: title for movie_id, title, *_ in reader}
	# Create a list of [Rating]
	with open(RATINGS, encoding="iso-8859-1") as f:
	    reader = csv.reader(f, delimiter="\t")
	    ratings = [Rating(user_id, movie_id, float(rating))
	               for user_id, movie_id, rating, _ in reader]
	
	# Split data
	import random
	random.seed(0)
	random.shuffle(ratings)
	
	split1 = int(len(ratings) * 0.7)
	split2 = int(len(ratings) * 0.85)
	
	train = ratings[:split1]              # 70% of the data
	validation = ratings[split1:split2]   # 15% of the data
	test = ratings[split2:]               # 15% of the data
```

_Explore Data_
```python
	import re
	
	# Data structure for accumulating ratings by movie_id
	star_wars_ratings = {movie_id: []
	                     for movie_id, title in movies.items()
	                     if re.search("Star Wars|Empire Strikes|Jedi", title)}
	
	# Iterate over ratings, accumulating the Star Wars ones
	for rating in ratings:
	    if rating.movie_id in star_wars_ratings:
	        star_wars_ratings[rating.movie_id].append(rating.rating)
	
	# Compute the average rating for each movie
	avg_ratings = [(sum(title_ratings) / len(title_ratings), movie_id)
	               for movie_id, title_ratings in star_wars_ratings.items()]
	
	# And then print them in order
	for avg_rating, movie_id in sorted(avg_ratings, reverse=True):
	    print(f"{avg_rating:.2f} {movies[movie_id]}")
```

_Code: Baseline Mode predicts the mean_
```python
	avg_rating = sum(rating.rating for rating in train) / len(train)
	baseline_error = sum((rating.rating - avg_rating) ** 2
	                     for rating in test) / len(test)
	
	# This is what we hope to do better than
	assert 1.26 < baseline_error < 1.27
```

_Code: Create User and Item Embedding (Random Initially) _
```python
	from scratch.deep_learning import random_tensor
	
	EMBEDDING_DIM = 2
	
	# Find unique ids
	user_ids = {rating.user_id for rating in ratings}
	movie_ids = {rating.movie_id for rating in ratings}
	
	# Then create a random vector per id
	user_vectors = {user_id: random_tensor(EMBEDDING_DIM)
	                for user_id in user_ids}
	movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM)
	                 for movie_id in movie_ids}
```

_Code: Training Loop_
```python
	from typing import List
	import tqdm
	from scratch.linear_algebra import dot
	
	def loop(dataset: List[Rating],
	         learning_rate: float = None) -> None:
	    with tqdm.tqdm(dataset) as t:
	        loss = 0.0
	        for i, rating in enumerate(t):
	            movie_vector = movie_vectors[rating.movie_id]
	            user_vector = user_vectors[rating.user_id]
	            predicted = dot(user_vector, movie_vector)
	            error = predicted - rating.rating
	            loss += error ** 2
	
	            if learning_rate is not None:
	                #     predicted = m_0 * u_0 + ... + m_k * u_k
	                # So each u_j enters output with coefficent m_j
	                # and each m_j enters output with coefficient u_j
	                user_gradient = [error * m_j for m_j in movie_vector]
	                movie_gradient = [error * u_j for u_j in user_vector]
	
	                # Take gradient steps
	                for j in range(EMBEDDING_DIM):
	                    user_vector[j] -= learning_rate * user_gradient[j]
	                    movie_vector[j] -= learning_rate * movie_gradient[j]
	
	            t.set_description(f"avg loss: {loss / (i + 1)}")
```

_Code: HyperParameter Tuning_
- Decrease learning rate
```python
	learning_rate = 0.05
	for epoch in range(20):
	    learning_rate *= 0.9
	    print(epoch, learning_rate)
	    loop(train, learning_rate=learning_rate)
	    loop(validation)
	loop(test)
```


# 24 Database and SQL
### 24.1 Create and Insert

### 24.2 Update


### 24.3 Delete

### 24.4 Select

### 24.5 Group By

### 24.6 Order By


### 24.7 Join


### 24.8 Query Optimization


### 24.9 NoSQL




# 25 MapReduce
### 25.1 Example: Word Count


### 25.2 Why MapReduce


### 25.3 MapReduce More Generally

### 25.4 Example: Analyzing Status Update


### 25.5 Example: Matrix Multiplication



### 25.6 Combiners
