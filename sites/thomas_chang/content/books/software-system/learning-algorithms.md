+++
title = "Learning Algorithms"
description = "Refresher on common software algorithms"
+++
# 1 Problem Solving

## 1.1 What Is An Algorithm
- The study of algorithms is concerned with both
    - correctness (will this algorithm work for all input?)
    - performance (is this the most efficient way to solve this problem?)
- Key Questions
    - What are the factors impacting an algorithm's performance?
    - How to Analyze algorithm performance independent of the implmentation?
- Terminologies
    - _problem instance_ = 1 input + 1 run of the algorithm

## 1.2 Finding The Largest Value In a List


## 1.3 Counting Key Operations
- How do you define the key operation?
- We use the key operation to compare 2 algorithms
    - Ex:
      ```python
      def alternate(A):
        for v in A:
          v_is_largest = True        
          for x in A:
            if v < x: # <-- KEY OPERATION !!
              v_is_largest = False   
              break
          if v_is_largest:
            return v	               
        return None       
      ```

## 1.4 Models Can Predict Algorithm Performance
- To quantify algo performance, find the ==inputs== that generates the
    - best case (ie least number of key operations)
    - worst case (ie most number of key operations)
- Example
    - For the alternate method above,
        - the best case is when input is ascending order
        - the worst case is when input is descending order
            - $n!$
        - key operation is the < operator
            - run time correlates to the < operator
              ![[Book-LearningAlgo-Ch1-BestWorstCase.png]]


- A common table to compare two algos
  | N      | Algo1_WorstCase | Algo2_WorstCase      | Algo1_BestCase | Algo2_BestCase     |
  | :---        |    :----:   | :---        |    :----:   |          ---: |
  | 8      | Title       | 8      | Title       | Here's this   |
  | 16   | Text         | 16   | Text        | And more      |
    -  Even though Big O is based on worst case


## 1.5 Find 2 Largest Values In A List
- Algo
```python
	def largest_two(A):
	  my_max,second = A[:2]                
	  if my_max < second:
	    my_max,second = second,my_max
	
	  for idx in range(2, len(A)):
	    if my_max < A[idx]:                
	      my_max,second = A[idx],my_max
	    elif second < A[idx]:              
	      second = A[idx]
	  return (my_max, second)
```
- Counting the number of times less-than is invoked is more complicated because, once again, it depends on the values in the problem instance.
- Best Case: Descending
    - 1 + (N-2) = N-1
- Worst Case: Ascending
    - we need to execute 2 < operators per element in array
    - 1 + 2x(N-1) = 2N-2
        - 1 is the for the first comparsion outside of the loop
- What other factors impact peformance?
    - required storage?  we introduce 2 new variables
    - programming effort? number of code lines --> read, maintain
    - mutable input
    - speed: does algorithm


## 1.6 Tournament Algorithm

## 1.7 Time and Space Complexity

_Time Complexity_
- The goal of time complexity is to come up with a function C(N) that counts ==the number of elementary operations== performed by an algorithm as a function of N, the size of a problem instance.
    - Previous section, we talked about key operations (ie < comparison). Here, we are focusing on elementary operations, ie addition
-  The structure of a program is critical to the time complexity
    - Constant: O(3)
      ```python
      def f0(N):
          cnt = 0
          cnt += 1
          cnt += 1
      ```
    - Linear: O(N)
      ```python
      def f1(N):
          cnt = 0
          for i in range(N):
              cnt += 1
          return cnt
      ```
    - Linear: O(4N)
      ```python
      def f2(N):
          cnt = 0
          for i in range(N):
              cnt += 1
              cnt += 1
              cnt += 1
              cnt += 1
          return cnt		```
    - Quadriatic: O($N^2$)
      ```python
      def f3(N):
          cnt = 0
          for i in range(N):
              for j in range(N):
                  cnt += 1
          return cnt
      ```



_Space Complexity_
- Space complexity accounts for extra memory required by an algorithm based on the size N of the input
    - memory can be file system or RAM.
    - In largest2() function above, we defined 2 variables my_max,second . The memory consumption is constant because it does not increase with the input size N.



## 1.8 Summary



# 2 Analyzing Algorithms
- This chapter discusses how to ensure the algo works for all inputs (ie correctness)



## 2.1 Using Empirical Models To Predict Performance

## 2.2 Multiplication Can Be Faster
- Throughout this book, I will describe the implementations of the algorithms, and ==based on the structure of the code==, I can identify the appropriate formula to model the performance of an algorithm.
-
## 2.3 Performance Classes
- There are a set of different performance classes
    - Sub Linear
        - Constant
        - O(1)
    - Polynomial
        - Linear: O(N)
        - O(N Log N)
        - O($N^2$)
    - Exponential
        - recursive algorithm ==can== be exponential
        - O($N^3$)
        - Exponential: O($2^n$)
            - Recursive algo solves a problem of size N by solving ==two== smaller problem of size N-1. Without maintaing state, we could end up with duplicate work.
            - Ex: Tower of Hanoi:
              ```python
          def solve_hanoi(N, sfrom_peg, to_peg, spare_peg) {
          if (N<1)
          return;
          solve_hanoi(N-1, from_peg, spare_peg, to_peg);
          solve_hanoi(N-1, spare_peg, to_peg, from_peg);
          }
          ```
        - Factorial: O(N!)
            - Generate all permutation of a list
                - Logically, you have $n!$ lists.  So at least O($n!$)
                - Permutation is position dependent.
- The goal is to find a model that predicts ==_the worst runtime performance (aka upper bound)_== for a given problem instance N.

## 2.4 Asymptotic Analysis
- You can try to throw advanced computing hardware at a problem, but eventually the more efficient algorithm will be faster for large-enough problem instances.
- To estimate the runtime performance for an algorithm on a problem instance of size N, start by counting the number of operations. It’s assumed that each operation takes a fixed amount of time to execute, which turns this count into an estimate for the runtime performance.

## 2.5 Counting All Operations
- Example
  ```python
  for i in range(100):
    for j in range(N):
        ct = ct + 1
  ```
    - O(N) = 100 x N --> O(N)


## 2.6 Counting All Bytes
- One can do something simlar with memory
  ```python
>>> import sys
>>> sys.getsizeof(range(100))
48
>>> sys.getsizeof(range(10000))
48
>>> sys.getsizeof(list(range(100)))
1008
>>> sys.getsizeof(list(range(1000)))
9112
>>> sys.getsizeof(list(range(10000)))
90112
>>> sys.getsizeof(list(range(100000)))
900112
```
- for range(100) and range(10000) --> O(1) constant because both consumes 48 bytes
- for range(10000) and range(100) --> O(N)


## 2.7 When One Door Closes, Another One Opens

## 2.8 Binary Array Search

## 2.9 Almost As Eash as $\pi$

## 2.10 Two Brids With One Stone

## 2.11 Putting It All Together

## 2.12 Curve Fitting Vs Lower/Upper Bounds

## 2.13 Summary


# 3 Better Living Through Better Hashing
## 3.1 Associating Values With Keys

- Example: Base 26
```python
	def base26(w):
	  val = 0
	  for ch in w.lower():                   
	    next_digit = ord(ch) - ord('a')      
	    val = 26*val + next_digit            
	  return val
```





## 3.2 Hash Functions and Hash Codes


## 3.3 A Hash Table Structure for (Key, Value) Pairs

```python
class Entry:
  def __init__(self, k, v):
    self.key = k
    self.value = v
    
```

## 3.4 Detecting And Resolving Collisions with Linear Probing
- One common approach, called _linear probing_, has `put()` incrementally check higher index positions in `table` _for the next available empty bucket_ if the one designated by the hash code contains a different entry; if the end of the array is reached without finding an empty bucket, `put()` continues to search through `table` starting from index position 0. This search is guaranteed to succeed _because I make sure to always keep one bucket empty_;

```python
	class Hashtable:
	  def __init__(self, M=10):
	    self.table = [None] * M
	    self.M = M
	    self.N = 0
	
	  def get(self, k):
	    hc = hash(k) % self.M           
	    while self.table[hc]:
	      if self.table[hc].key == k:   
	        return self.table[hc].value
	      hc = (hc + 1) % self.M        
	    return None                     
	
	  def put(self, k, v):
	    hc = hash(k) % self.M           
	    while self.table[hc]:
	      if self.table[hc].key == k:   
	        self.table[hc].value = v
	        return
	      hc = (hc + 1) % self.M        
	
	    if self.N >= self.M - 1:        
	      raise RuntimeError ('Table is Full.')
	
	    self.table[hc] = Entry(k, v)    
	    self.N += 1
```

## 3.5 Separate Chaining With Linked List (LL)

```python
	class LinkedEntry:
	  def __init__(self, k, v, rest=None):
	    self.key = k
	    self.value = v
	    self.next = rest  
	
	class Hashtable:
	  def __init__(self, M=10):
	    self.table = [None] * M
	    self.M = M
	    self.N = 0
	
	  def get(self, k):
	    hc = hash(k) % self.M       
	    entry = self.table[hc]      
	    while entry:
	      if entry.key == k:        
	        return entry.value
	      entry = entry.next
	    return None
	
	  def put(self, k, v):
	    hc = hash(k) % self.M       
	    entry = self.table[hc]      
	    while entry:
	      if entry.key == k:        
	        entry.value = v         
	        return
	      entry = entry.next
	
	    self.table[hc] = LinkedEntry(k, v, self.table[hc])  
	    self.N += 1
```


## 3.6 Removing an Entry From LL
```python
	def remove(self, k):
	  hc = hash(k) % self.M
	  entry = self.table[hc]             
	  prev = None
	  while entry:                       
	    if entry.key == k:               
	      if prev:
	        prev.next = entry.next       
	      else:
	        self.table[hc] = entry.next  
	
	      self.N -= 1                    
	      return entry.value
	
	    prev, entry = entry, entry.next  
	
	  return None
```


## 3.7 Evaluation


## 3.8 Growing HashTables

## 3.9 Analyzing the Performance of Dynamic Hashtables

## 3.10 Perfect Hashing

## 3.11 Iterate Over (key, value) Pairs

## 3.12 Summary



# 4 Heaping It On
- Priority queue stores entries with values, where priority is associated with the value.
    - a data type that efficiently supports enqueue(value, priority) and dequeue(), which removes the value with highest priority.

## 4.1 Max Binary Heaps
- Max ==binary== heap partially sort the entries; only the max.
    - enqueue and dequeue has ==O(log N)== performance (by maintaining a tree structure)
    - binary because there is a left and right child

- Use python heapq module
```python
	import heapq # implements min heap
	
	data = []
	heapq.heapify(data)
	
	smallest = heapq.heappop(data) #O(1)
	
	heapq.heappush(10) # O(log(N))

```

## 4.2 Inserting a (value, priority)


## 4.3 Removing the Value with Highest Priority

## 4.4 Representing a Binary Heap In a Array

## 4.5 Implementation of Swim and Sink

## 4.6 Summary




# 5 Sorting Without A Hat
## 5.1 Sorting By Swapping

## 5.2 Selection Sort

## 5.3 Anatomy of a Quadriatic Sorting Algorithm

## 5.4 Analyze Performance of Insertion and Selection Sort

## 5.5 Recursion and Divide and Conquer

## 5.6 Merge Sort


## 5.7 Quick Sort


## 5.8 Heap Sort


## 5.9 Performance Comparison of O(N log N)

## 5.10 Time Sort



# 6 Binary Trees: Infinity In the Palm of Your Hand
## 6.1 Getting Started
- Binary tree, like linked list, is a ==recursive== data structure
- Ex 1: Linked List recursive structure enables recursive sum computation
```python
	class Node:
	  def __init__(self, val, rest=None):
	    self.value = val
	    self.next = rest
	
	def sum_iterative(n):
	  total = 0                         
	  while n:
	    total += n.value                
	    n = n.next                      
	  return total
	
	def sum_list(n):
	  if n is None:                     
	    return 0
	  return n.value + sum_list(n.next)
```

- Example 2: Binary trees for expressions
```python
	class Value:                                    
	  def __init__(self, e):
	    self.value = e
	
	  def __str__(self):
	    return str(self.value)
	
	  def eval(self):
	    return self.value
	
	class Expression:                               
	  def __init__(self, func, left, right):
	    self.func  = func
	    self.left  = left
	    self.right = right
	
	  def __str__(self):                            
	    return '({} {} {})'.format(self.left, self.func.__doc__, self.right)
	
	  def eval(self):                               
	    return self.func(self.left.eval(), self.right.eval())
	
	def add(left, right):                           
	  """+"""
	  return left + right

	a = Expression(add, Value(1), Value(5))
	m = Expression(mult, a, Value(9))
	print(m, '=', m.eval())
	((1 + 5) * 9) = 54
```
## 6.2 Binary Search Trees
- Has efficient $log(N)$ search, insert, and remove operations

```python
class BinaryNode:
  def __init__(self, val):
    self.value = val             
    self.left  = None            
    self.right = None    


class BinaryTree:
  def __init__(self):
    self.root = None                             

  def insert(self, val):                         
    self.root = self._insert(self.root, val)

  def _insert(self, node, val):
    if node is None:
      return BinaryNode(val)                     

    if val <= node.value:                        
      node.left = self._insert(node.left, val)
    else:                                        
      node.right = self._insert(node.right, val)
    return node    
    
```

## 6.3 Searching for Values for a BST
```python
class BinaryTree:
  def __contains__(self, target):
    node = self.root                 
    while node:
      if target == node.value:       
        return True

      if target < node.value:        
        node = node.left
      else:
        node = node.right            

    return False      
```


## 6.4 Removing Values for a BST
```python
	def _remove_min(self, node):
	  if node.left is None:                    
	    return node.right
	
	  node.left = self._remove_min(node.left)  
	  return node          
```


```python
	def remove(self, val):
	  self.root = self._remove(self.root, val)         
	
	def _remove(self, node, val):
	  if node is None: return None                     
	
	  if val < node.value:
	    node.left = self._remove(node.left, val)       
	  elif val > node.value:
	    node.right = self._remove(node.right, val)     
	  else:                                            
	    if node.left is None:  return node.right
	    if node.right is None: return node.left        
	
	    original = node                                
	    node = node.right
	    while node.left:                               
	      node = node.left
	
	    node.right = self._remove_min(original.right)  
	    node.left = original.left                      
	
	  return node
```


## 6.5 Traversing a Binary Tree
- In order traversal
```python
	class BinaryTree:
	
	  def __iter__(self):
	    for v in self._inorder(self.root):      
	      yield v
	
	  def _inorder(self, node):
	    if node is None:                        
	      return
	
	    for v in self._inorder(node.left):      
	      yield v
	
	    yield node.value                        
	
	    for v in self._inorder(node.right):     
	      yield v

```


## 6.6 Analyzing Performance of BST

## 6.7 Self Balancing Binary Tree

## 6.8 Analyzing Performance of Self Balancing Trees

## 6.9 Using Binary Tree as (key, value) Symbol Table

## 6.10 Using the Binary Tree as a Priority Queue





# 7 Graphs: Only Connect

## 7.1 Graphs Efficiently Store Useful Information


## 7.2 Using Depth First Search to Solve a Maze


## 7.3 Breadth First Search (BFS) Offers Different Searching Strategy


## 7.4 Directed Graphs


## 7.5 Graphs with Edge Weights


## 7.6 Dijkstra's Algorithm


## 7.7 All Pairs Shortest Path


## 7.8 Floyd Warshall Algorithm

