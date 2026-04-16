+++
title = "Scaling ML Spark"
description = "Refresher course on Spark"
+++


# 1 Distributed ML Terms and Concepts


# 2 Intro to Spark and PySpark
- Driver Program
    - The driver program (aka Spark driver) is a dedicated process that runs on the driver machine. It is responsible for executing and holding the SparkSession, which encapsulates the SparkContext—this is considered the application’s entry point, or the “real program.” The SparkContext contains all the basic functions, context delivered at start time, and information about the cluster. The driver also holds the DAG scheduler, task scheduler, block manager, and everything that is needed to turn the code into jobs that the worker and executors can execute on the cluster. The driver program works in synergy with the cluster manager to find the existing machines and allocated resources.
- Executor:
    - Is a process
    - Has T tasks
        - tasks is smallest unit of work
        - task in the same executor share the same cache, memory, and global parameters
- Worker Node
    - VM
    - multiple executor on a worker node
- Cluster manager
    - Assign executors to worker nodes, assign resources and communicate information about resource availability to driver program

### Executing PySpark Code
- Pyspark ->
    - process: python - InterProcess communication (IPC) - process: scala/JVM
    - JVM spark query optimization, computation, distribution of tasks to clusters
- Py4J
    -  pyspark driver - pickled item (serialization) - py4j context - Spark driver GVM
    -  [Py4J](https://www.py4j.org/) is a library written in Python and Java that enables Python programs running in a Python interpreter to dynamically access Java objects and collections in a JVM via standard Python methods
    - ==PySpark is often less efficient in terms of running time than traditional Scala/Java code==

### Apache Spark Basics
- Architectures
- DataFrame vs Datasets
    - Datasets enforces type safety; there is no way to mistaken a col type of string as int
    - BUT dataset is slower.



# 3 MLFlow
- MLOps: manage the machine learning experiment lifecycle
    - term: experiments can be orgainzed as pipeline code or Ml modules in repos
    - Phases
        - machine learning
            - training, tuning, finding optimal models
        - development
            - develop ==pipelines== and ==tools== to take ml model from dev/experiment to staging and production
        - operations
            - CI/CD, monitoring, managing, serving models at scale

- Requirement for ML Lifecycle Management
    - Reproducibility
        - Ability to run your algorithm repeatedly on different datasets and obtain the same (or similar) results. Analyzing, reporting, auditing, and interpreting data are all aspects of this process, meaning each step in the lifecycle needs to be tracked and saved
    - Code Version Control
    - Data Version Control
        - Tools: DVC and lakeFS

- ML Flow manages the entire ML lifeycle
    - Component 1: Tracking Server
        - What does it track?
            - Model hyper-parameters
            - metrics
            - metadata
            - artifacts
        - Runs: everytime you run an experiment, you can tack
        - `$ mlflow run example/conda-env/project -P alpha=0.5`
    - Compoment 2:  Model Registry
        - Downstream automated job can query the model registry
        - DS -> model into staging
        - ML Engineer -> production and archived
- Tbale schema
    - Experiment -> Run
    - Run -> Metrics, Params, and Tags

# 4 Data Ingestion, PreProcessing, and Stats


# 5 Feature Engineering
### Terminology
- Estimator : an algorithm that can fit on a Dataframe

### Common Scenarios and Options
Handle Missing Features
- Approaches
    - drop rows
    - mean/median
    - random sample
    - arbitrary value
    - missing value indicator
    - multivariate imputation
- Some algorithms are more sensitive to missing values, like SVM, PCA and nearest neighbor because each data point changes the outcome.

Extract Feature From Text
- BOW
- TF-IDF
- n-gram
- word2vec
- Topic extraction

Categorical Encoding
- StringToIndex
- 1 hot encoding
- Count and frequency encoding
- Ordinal encoding
- Weight of eveidence
- Rare label encoding
- Feature hashing

Feature Scaling
- Normalization
- MinMax scaling
- Mean scaling
- Max absolute scaling
- Unit norm scaling

# 6 Training Models with Spark MLLib
- Spark model can view its hypermeter documentation
```python
	import pprint 
	pp = pprint.PrettyPrinter(indent=4)
	params = model.explainParams() 
	pp.pprint(params)

('aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)\n'
 'featuresCol: features column name. (default: features, current: '
 'selectedFeatures)\n'
 'k: Number of independent Gaussians in the mixture model. Must be > 1. '
 '(default: 2, current: 42)\n'
 ....
)
```

### 6.1 Supervised ML
#### 6.1.1 Classification
- classification calculates the probability of a data point belong to a class(es) given the input features. Output is a probability that is mapped/thresholded to categorical.
    - Distinction
        - logistic regression outputs the probability of a discrete class (used in binary classfication). LR is classifcation
        - regression algorithms predictions continuous values
- Types of classification
    - Binary
        - Each input is classified into one of two classes (yes or no, true or false, etc.).
    - Multiclass
        - Each input is classified into one of a set of more than two classes.
    - Multilabel
        - Each given input can have multiple labels in practice. For example, a sentence might have two sentiment classifications, such as happy and fulfilled. Spark does not support this out of the box; you will need to train each classifier separately and combine the outcomes.
- Algorithms
    - Logistic Regression:
        - Binary and multiclass classifier. Can be trained on streaming data with the RDD-based API. Expects an indexed label and a vector of indexed features.
    - Decision Tree Classifier
        - Binary and multiclass decision tree classifier. Expects an indexed label and a vector of indexed features.
    - RandomForest Classifier
        - Binary and multiclass classifier. A random forest is a group or ensemble of individual decision trees, each trained on discrete values. Expects an indexed label and a vector of indexed features.
    - GBTClassifier
        -   Binary gradient boosted trees classifier (supported in Spark v3.1.1 and later). Like the `RandomForestClassifier`, this is an ensemble of decision trees. However, its training process is different; as a result, ==it can be used for regression as well==. Expects an indexed label and a vector of indexed features.
    - MultiLayer Perception
        - Multiclass classifier based on a feed-forward artificial neural network. Expects layer sizes, a vector of indexed features, and indexed labels.
    - OneVsRest
        - Used to reduce multiclass classification to binary classification, using a one-versus-all strategy. Expects a binary classifier, a vector of indexed features, and indexed labels.
    - Naives Bayes
        - Multiclass classifier, considered efficient as it runs only one pass over the training data. Expects a Double for the weight of a data point (to correct for a skewed label distribution), indexed labels, and a vector of indexed features. Returns the probability for each label.
    - FMClassifier
        - Binary factorization machines classifier. Expects indexed labels and a vector of indexed features.
- Sample code
```python
from pyspark.ml.classification import LogisticRegression

happy_lr = LogisticRegression(maxIter=10, labelCol="happy_label")
happy_lr_model = happy_lr.fit(train_df)
```

- How to deal with imbalanced data?
    - Effect: Bias will be toward teh class label with high number of observation bc it is statistically more dominant
    - Sol 1:  downsize the majority and upsize the minority
    - Sol 2: Algorms it GBTClassifier, GBTRegression, and RandomForestClassifier has a [featureSubsetStrategy](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html#pyspark.ml.classification.RandomForestClassifier.featureSubsetStrategy). In every tree node, the algorithm processes a random subset of features and uses the result to build the next node

#### 6.1.2 Regression
- Types of Regression
    - Simple:
        - There is only one independent and one dependent variable: one value for training and one to predict.
    - Multiple
        - Here we have one dependent variable to predict using multiple independent variables (features) for training and input.
    - Multivariate
        - Similar to multilabel classification, there are multiple variables to predict (labels) with multiple independent variables for training and input. Accordingly, the input and output are vectors of numeric values.
- Algorithms [spark](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html#regreschang

### 6.2 Unsupervised

#### 6.2.1 Frequent Pattern Mining
- A type of association learning
  - based on identifying rules to uncover relationships between variables in the data. Association rule mining algorithms typically first look for frequent items in the dataset, then for frequent pairs or itemsets (e.g., items that are often viewed or purchased together). The rules follow the basic structure of antecedent (if) and consequent (then).
- Algorithms
  - FPGrowth
  - PrefixSpan: (MCKernel!)


#### 6.2.2 Clustering
- Common concepts
    - k = num clusters
    - some algo has a weightCol in trainingData that represents the data point's importance relative to the cluster center
- Algorithms
    - LDA
        - LDA (Latent Dirichlet Allocation) is a generic statistical algorithm used in evolutionary biology, biomedicine, and natural language processing. It expects a vector representing the counts of individual words in a document; since in our scenario we’re focusing on variables like fuel type, LDA does not match our data.
    - GauusianMixture
        - The GaussianMixture algorithm is often used to identify the presence of a group within a bigger group. In our context, it could be useful for identifying the subgroups of different classes inside each car manufacturer’s group, such as the compact car class in the Audi group and the Bentley group. However, Gaussian​Mixture is known to perform poorly on high-dimensional data, making it hard for the algorithm to converge to a satisfying conclusion. Data is said to be high-dimensional when the number of features/columns is close to or larger than the number of observations/rows. For example, if I have five columns and four rows, my data is considered high-dimensional. In the world of large datasets, this is less likely to be the case.
    - KMeans
        - KMeans is the most popular algorithm for clustering, due to its simplicity and efficiency. It takes a group of classes, creates random centers, and starts iterating over the data points and the centers, aiming to group similar data points together and find the optimal centers. The algorithm always converges, but the quality of the results depends on the number of clusters (k) and the number of iterations.
    - PowerIterationClustering
        - `PowerIterationClustering` (PIC) implements the [Lin and Cohen algorithm](https://oreil.ly/-Gu9c). It’s a scalable and efficient option for clustering vertices of a graph given pairwise similarities as edge properties. Note that this algorithm cannot be used in Spark pipelines as it does not yet implement the `Estimator`/`Transformer` pattern (more on that in [“Machine Learning Pipelines”](https://learning.oreilly.com/library/view/scaling-machine-learning/9781098106812/ch06.html#ml_pipelines)).
- Sample code (Gaussian mixture)
```python
	k = dataset.select("Make").distinct().count()
	
	from pyspark.ml.clustering import GaussianMixture
	gm = GaussianMixture(k, tol=0.01, seed=10,                       featuresCol="selectedFeatures", maxIter=100)
	model = gm.fit(dataset)
	
	summary = model.summary 
	# summary contains the predicted cluster centers, the transformed predictions, the cluster size (i.e., the number of objects in each cluster), and dedicated parameters based on the specific algorithm

	summary.clusterSizes
	summary.logLikelihood
	
```



### 6.3 Evaluation
- Spark has 6 different evaluators, each implementing the abstract class Evaulator

#### 6.3.1 Supervised
- First calculates the confusion matrix

  		Predicted
  ACTUAL		 -  TP      FN
  -  FP     TN

- Types
    - BinaryClassicationEvaluator
    - MulticlassClassicationEvaluator
    - MultilabelClassicationEvaluator
    - RegressionEvaluator
    - RankingEvaluator


#### 6.3.2 Unsupervised
- Computes the silhouette measure
    - It expects two input columns, prediction and features, and an optional weight column. It computes the Silhouette measure, where you can choose between two distance measures: squaredEuclidean and cosine. ==Silhouette is a method to evaluate the consistency and validity of the clusters==; it does this by calculating the distance between each data point and the other data points in its cluster and comparing this to its distance from the points in other clusters.

- Code
```python
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator(featuresCol='selectedFeatures')
evaluator.setPredictionCol("prediction")

# compare kmeans vs gm
print("kmeans: "+str(evaluator.evaluate(kmeans_predictions)))
print("GM: "+ str(evaluator.evaluate(gm_predictions)))
	kmeans: 0.7264903574632652
	GM: -0.1517797715036008

evaluator.isLargerBetter() # true or false

# sets the distance measurement
evaluator.setDistanceMeasure("cosine")
print("kmeans: "+str(evaluator.evaluate(kmeans_predictions)))
print("GM: "+ str(evaluator.evaluate(gm_predictions)))

```

### 6.4 Hyperparameter and Tuning Experiments
#### Option 1: Parameter Grid
```python

# Build grid
from pyspark.ml.tuning import ParamGridBuilder

grid = ParamGridBuilder()
		.addGrid(kmeans.maxIter, [20,50,100]) 
        .addGrid(kmeans.distanceMeasure, ['euclidean','cosine']) 
        .addGrid(evaluator.distanceMeasure, ['euclidean','cosine']).build()    

# Split data into training and test
from pyspark.ml.tuning import TrainValidationSplit
tvs = TrainValidationSplit(estimator=kmeans, estimatorParamMaps=grid, 
                           evaluator=evaluator, collectSubModels=True, seed=42)
tvs_model = tvs.fit(data)

# See results
tvs_model.validationMetrics

```


#### Option 2: Cross Validation
- Cross validation enables use to try different multiple combiation of data splits; BUT will be compute intensive
```python
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=kmeans, estimatorParamMaps=grid, 
                    evaluator=evaluator, collectSubModels=True, 
                    parallelism=2, numFolds=3)
cv_model = cv.fit(data)
```


### 6.5 ML Pipellines
- Spark Concepts
    - Transformers = A function that converts data in some wa
    - Estimator = an algorithm
    - Pipeline = sequence of transformer and estimator
```python
	from pyspark.ml import Pipeline

	# transformer: hasher, selector
	# estimator: gm
	pipeline = Pipeline(stages=[hasher, selector, gm])
	
	# Fit the pipeline to training data
	model = pipeline.fit(data)
```


# 7 Bridging Spark and DL Frameworks
### 2 Cluster Approach
- Spark philosophy makes it ideal to data processing and feature engineering
    - ==`the size of the cluster changes the duration of the job, not its capacity to run an algorithm`==
    - Spark’s limitations are mostly bound to its underlying premise that all algorithm implementations must be able to scale without limit, which requires the model to be able to perform its learning process at scale
- In a two cluster approach
    - cluster 1: Spark cluster
    - cluster 2: DL distributed cluaster
    - (cluster1: Spark) <-- (S3, HDFS, GCP storage) --> (cluster2: DL)

- Monoids
    - A monoid is an algebraic structure that includes a set, an associative binary operation, and an identity element
      From a programming standpoint, monoids enable developers to break up a task into smaller subtasks almost arbitrarily,
        - So what? Break up enables us to decouple logic.  One use case is to delay the error handling to a common piece of code, Either error handling pattern

### Data Access Layer
- Why need Data Access Layer (DAL)
    - If we use the best toool for each
        - Spark for data processing, feature engineering
        - Torch & Tensorflow for distributed training
    - we need to have a way to transfrom between the different data formats

- DAL feature requirements
    - Distributed systems:
        - It should be able to leverage existing systems to enable scaling.
    - Rich software ecosystem:
        - This allows it to grow to encompass new machine learning frameworks and ensures that it will continue to be supported, bugs will be fixed, and new features will be developed.
    - Columnar file formats:
        - Columnar file formats store data by column, not by row, which allows greater efficiency when filtering on specific fields during the training and testing process.
    - Row filtering
        - Some machine learning algorithms require sampling specific rows, so we need a mechanism to filter on rows, not only on columns.
    - Data versioning
        - It should be possible to travel back in time, to support reproducibility of experiments.

- Author discusses open source Petastorm as DAL


# 8 Tensorflow Distributed Machine Learning
- This section relates to (distributed) training


# 9 Pytorch Distributed Machine Learning
- This section relates to (distributed) training


# 10 Deployment Patterns
### 10.1 Deployment Patterns



### 10.2 Monitoring ML Models In Production
- Data Drift
    - Definition:
        - the data you’re feeding to the online algorithm has changed in some way relative to the training data
    - Can occur any time the data structure, semantics, or infrastructure change unexpectedly
    - Modes
        - Instantaneous
        - Gradual
        - Periodic : Ebay's weekend traffic is different from weekdays
        - Temporary Drift
- Model Drift, Concept Drift
    - Changes in real-world environments often result in model drift, which diminishes the predictive power of the model. Many things may cause this, ranging from changes in the digital environment (leading to changes in relationships between model variables) to changes in user demographics or behaviors.
    - Detection of model drift is known to be a hard problem and often requires human interaction. It’s important to work with users, customers, product managers, and data analysts to better understand the changes that may affect the usefulness of the model
- Distributional Domain Shift (the long tail)
    - Domain shift refers to a difference between the distribution in the training dataset and the data the model encounters when deployed
    - Another way to look at it is to acknowledge that the sampling data may not represent all the parts of the distribution that we care about.
    - A domain shift can happen because of bugs in the training data pipeline or bias in the sampling proces
    - distribution of the training data does not accurately represent real-world data anymore
    - ==watch for differences between the training distribution and the production data distribution==
- What Metrics To Monitor
    - Model Metrics
        - accuracy, robustness, performance
        - May be hard to collect in proudction since we may not have all the required data
            - Ex: if supervised model, what is our label?
    - Business Metrics
        - These show the impact of the machine learning system on the business. For example, for a recommendation system, we would monitor various metrics related to user churn and user engagement: how many people are using the system, how frequently, and for how long in each interaction. We can even split the user base into multiple user groups and run A/B testing of different models in production. Monitoring business metrics is often fairly straightforward, as many organizations already have a business intelligence (BI) or analytics team that measures them. However, there may be conflicting or hidden factors that affect these metrics, so it’s best to combine these with other measures
    - Model prediction vs actual behavior
        - These metrics show how well the model’s predictions correlate with actual user or system behavior. Measuring them typically requires some creativity and tailor-made solutions, as it involves capturing real behaviors rather than predicted ones. Often, we will want to create a separate data pipeline to capture and save actual behavior into a dataset
    - Hardware/networking metrics
- How Do I Measure Changines Using My Monitoring System?
    - Goal: detect changes over time
    - Define a reference to compare to
        - validation and training data
        - production window over a period you believe to be healthy
    - Measure the reference over a time period
    - Algorithms for measurement
        - Rule based distance metrics
            - These measure how far the data is from the reference, given a set of rules. They’re great for determining the quality of the data. We can compare minimum, maximum, and mean values and check that they are within the acceptable/allowable range. We can also check the number of data points to confirm data isn’t missing or being dropped (for example, due to bad preprocessing), check for the presence of null values, and monitor for data drift.
        - D1 distance
            - This is a classic distance metric that calculates the sum of the distances between the fixed data values. It’s easy to interpret and simplifies monitoring in general.
        - Kolmogorov-Smirnov statistic
            - This finds the distance between the empirical and cumulative distribution functions. It’s a commonly used metric that is relatively easy to interpret and plot on a chart.
        - Kullback-Leibler Divergence
            - This measures the difference between two probability distributions over the same variable x. It’s a statistical log-based equation that is sensitive to the tails of the distribution. It detects outliers but is a bit difficult to comprehend and interpret. This metric can be useful when you’re fully informed about how to use it, but it won’t provide much insight in most cases.
- What it look like in production
    - Data check
        -  reference vs recent data --> data eval --> drift? --> retrain
    - Model
        - prediction vs actual --> model eval --> drifit? --> retrain


### 10.3 Production Feedback Loop
- A feedback loop is when the system saves the outputs of the model and the corresponding end user actions as observed data and uses this data to retrain and improve the model over time
- [UBER Paper](https://arxiv.org/pdf/2104.00087.pdf)


### 10.4 Deploying with MLLib
- Log MLLib models parameters
``` scala
	
	# Metadata
	("timestamp" -> System.currentTimeMillis())
	("sparkVersion" -> sc.version)
	("uid" -> uid)
	("paramMap" -> jsonParams)
	("defaultParamMap" -> jsonDefaultParams)
	
	# Model params
	"numFeatures"
	"numClasses"
	"numTrees"

```


### 10.5 Deploying with MLFlow

#### 10.5.1 Step1: Wrap model and meta in a MLflow Wrapper
	- Code 1: python wrapper code needs to implement `mlflow.pyfunc.PyFuncModel`
``` python
class MyModelWrapper():
	def __init__(self, model_path):
	    self.model_path = model_path

	def load_context(self, context):
	    log(self.model_path)
	    self.model = mlflow.keras.load_model(model_uri=self.model_path)

	def predict(self, context, model_input):
	    import tensorflow as tf
	    import json
	
	    class_def = {
	        0: '212.teapot', 
	        1: '234.tweezer', 
	        2: '196.spaghetti', 
	        3: '249.yo-yo', 
	    }
	
	    rtn_df = model_input.iloc[:,0:1]
	    rtn_df['prediction'] = None
	    rtn_df['probabilities'] = None
	
	    for index, row in model_input.iterrows():
	        # resize and reshape the image
	        image = np.round(np.array(Image.open(row['origin']).resize((224,224)),
	                                  dtype=np.float32))
	        img = tf.reshape(image, shape=[-1, 224, 224, 3])
	      
	        # predict
	        class_probs = self.model.predict(img)
	
	        # take the class with the highest probability
	        classes = np.argmax(class_probs, axis=1)
	        class_prob_dict = dict()
	
	        # calculate probability for each class option:
	        for key, val in class_def.items():
	            class_prob_dict[val] = np.round(np.float(class_probs[0][int(key)]), 
	                                                     3).tolist()
	
	        rtn_df.loc[index,'prediction'] = classes[0]
	        rtn_df.loc[index,'probabilities'] = json.dumps(class_prob_dict)
	
	    return rtn_df[['prediction', 'probabilities']].values.tolist()

```


	- Code 2: to save model
``` python
	model_path = ".../mlruns/{experiment_id}/{run_id}/artifacts/models"
	wrappedModel = {some_class}(model_path)
	mlflow.pyfunc.log_model("pyfunc_model_v2", python_model=wrappedModel)
	
	# Output
		--- 58dc6db17fb5471a9a46d87506da983f
		------- artifacts
		------------ model
		------------ MLmodel
		------------- conda.yaml
		------------- input_example.json
		------------- model.pkl
		------- meta.yaml
		------- metrics
		------------ training_score
		------- params
		------------ A
		------------ ...
		------- tags
		------------ mlflow.source.type
		------------ mlflow.user
```


#### 10.5.2 Option1: Deploy as a model service
``` python
	# load model as a service
	model_path = ".../mlruns/{experiment_id}/{run_id}/artifacts/models"
	model = mlflow.pyfunc.load_model(model_path)
	model.predict(model_input)

```



#### 10.5.3 Option2: Dploy as Spark UDF
```python
	# Load model as a Spark UDF
	loaded_model = mlflow.pyfunc.spark_udf(spark, mlflow_model_path, 
	                                       result_type=ArrayType(StringType()))
	# Predict on a Spark DataFrame
	scored_df = (images_df
             .withColumn('origin', col("content"))
             .withColumn('my_predictions', loaded_model(struct("origin")))
             .drop("origin"))
```

