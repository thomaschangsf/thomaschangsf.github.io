+++
title = "Machine Learning Interview"
description = "Case Study of Various Machine Learning Systems"
+++


-----------------------------------------------
# 1 Clarify Requirement
-----------------------------------------------
- Chapter 2: Visual Search system
- Chapter 3: Google Street View Blurring system
- Chapter 4: Youtube Video Search
- Chapter 5: Harmful Content Detection
- Chapter 6: Video Recommendation Service
- Chapter 7: Event Recommendation Service
- Chapter 8: Ad Click Prediction
- Chapter 9: Similar Listings on Vacation Rental
- Chapter 10: Personalized News Feed
- Chapter 11: People You May Know


-----------------------------------------------
# 2 Frame as ML
-----------------------------------------------
### Defining the ML Objective
- Maximize # user clicks
- Maximize # completed videos
- Maximize watch time

### Specify Input and Output
- Late vs Early Fusion (Ch5)


### ML Category
- High Level
    - Supervised
        - Classification
            - Single Class (aka Binary)
                - Granularity matters: in detecting harmful content, if we have a binary class (harmful/safe), it will be hard to (1) explain why post is taken down (2) improve model because which we don't know which class performs less well
            - Multi-class
                - if the classes are mutually exclusive
            - Multi-label
                - if the classes are not mutually exclusive
            - Multi-task (Ch 5: Detecting Harmful Content)
                - multiple models learn multiple task simultaneously; each task has a loss function we are learning
                - Classification head : is a linear layer which projects X dimension to Y dimension to produce logits of size K, where K is the number of classes [hf blog](https://discuss.huggingface.co/t/what-is-the-classification-head-doing-exactly/10138/11).  This is then fed to a
                    - softmax for multiclass single label
                    - sigmoid for binary or (multi class multilabel) (book Deep Learning with Python p114)

          ```mermaid
          graph TD;
              SharedLayers-->FusedFeatures;
              ViolenceClassHead-->SharedLayers;
              NudityClassHead-->SharedLayers;
              HateClassHead-->SharedLayers;
         ```
		- Regression
		- Note: Most examples in book are supervised classification
	- Unsupervised
		- Clustering
		- Dimension Reduction
		- Association
	- Reinforcement Learning: 
		- AI agent learns to take actions in an environment in order to maximize a particular reward
	- Generative
		- Generates new data, rather than making predictions based on existing data
- Ranking
    - Statistical (Similarity Measurement) [ref](https://wuciawe.github.io/machine%20learning/math/2016/06/09/notes-on-similarity-measurements.html)
        - euclidean distance
        - cosine similarity
        - Ochiai coefficient
        - taniomoto metric
        - Jaccard index
        - Pearson coefficient
        - Spearman's rank correlation coefficient
        - Hamming distance
    - (Ch2: Representation Learning)
    - Visual Search (Ch 4)
        - Representation Learning
        - Video Encoder
        - Text Encoder
    - Text Search (Ch 4)
        - Inverted Index
    - Learning To Rank (Ch7)
        - Pointwise (Ch7,8,10)
        - Pairwise
        - Listwise
    - Session Based Recommendation
- Fusing Layers
- Classification
    - Single binary classifier (Ch5)
    -  One binary classifier per class  (Ch5)
    - Multi-label classifier (Ch5)
    - Multi-task classifier (Ch5)
        - Shared layer
        - Task Specific Layer
    - Non Personalized
        - Rule based
- Personalization (Ch6)
    - Content based filtering
    - Collaborative Filtering
    - Hybrid filtering
- Rule Based (Ch6, 7)
- Embedding Based (Ch7)


-----------------------------------------------
# 3 Data Preparation
-----------------------------------------------
### 3.1 Data Engineering
_Data Sources_
- Is it clean?
- Can it be trusted?
- Are there privacy issues?

_Data Storage_
- SQL: Relational (Postgres)
- NoSQL: KeyValue(Redis, DynamoDB), Column(Cassandra, HBase), Graph(Neo4J), Document(MongoDB, DynamoDB)

_Data Types_
- Structured
- Ex: Numerical & Categorical
- Resides in Relational, NoSQL, DataWarehouse
- Models: Decision Tree, Linear Models, SVM, KNN, Naives Bayse, and Deep Learning
- Unstructured
- Ex: Audio, Video, Image, Text
- Resides in NoSQL DB, Data Lakes
- Models: Deep Learning

_Entities_
- Images
- Users
- Posts
- Events
- Friendships
- Ads
- Listings
- EntityX-EntitY Interaction
- Annotated dataset


### 3.2 Feature Engineering
_Feature Engineering Operations_
- Handle Missing Values
- Replace with:
- default values
- mean, media, or mode (the most common value)
- Feature Scaling
- Rationale: Some ML models learn better when features values are in similar ranges
- Normalization: (aka min-max scaling)$$ z = \frac{x-x_{min} }{x_{max}-x_{min}} $$
- changes distribution
- Z-score normalization: $$ z = \frac{x-\mu }{\sigma} $$
- log scaling: good for long tail distribution
- Discretization (Bucketing)
- Encoding Categorical Features (1 hot encoding, )
- via embeddings: good for high cardinality features


- Preparing Text Data (Ch4)
    - Text Normalization
        - lower case
        - punctuation removal
        - trim white spaces
        - Lemmatization
        - Stemming
    - Tokenization
        - Word
        - Subword
        - Character
    - Inverted Index
    - Token To Ids
        - Lookup table
        - hashing
    - Text Encoder (training this can be considered as modeling)
        - Statistical: BOW, TFIDF
        - ML Based: (Ch5)
            - Embedding Layer: f(tokenId) => embedding
            - Word2Vec: use shallow neural network leverage co-occurrence of words in ==local context==
                - Continuous Bag of Word (cBOW)
                - Skipgram
            - Transformer: Bert, GPT3, Bloom
- Preparing Video Data (Ch2, 4)
    - Decoding,Sample
    - Resize, scale
    - z-score normalization
    - Correct Color Mode
    -
- User reaction to post (Ch5)
- Author Features (Ch5)
- Contextual Information (Ch5)
    - time of day
    - device
- Video Features (Ch6)
    - Video ID
    - Duration
    - Language
    - Titles and tags
- User Features (Ch6)
    - Demographics
    - Contextual
    - Historical interactions
        - Search history
        - Liked Videos
        - Watched Videos
        - Impressions
- Location Features (Ch7)
    - walk score
    - country
    - user to location affinity
- Time Related Features (Ch7)
    - time remaining till event
- Social Related Features (Ch7)
    - number of people attending
- Event Related Features (Ch7)
- Ad Features
    - Category/Subcategory
    - Engagement numbers
    - Image/Videos

-----------------------------------------------
# 4 Model Development
-----------------------------------------------
### 4.1 Model Selection
_Factors to consider in selecting models_
- Amount of data the model needs
- Training speed
- Hyper-parameters to choose
- Possibility of continuous/lifelong learning
- learn a model for a large number of tasks sequentially without forgetting knowledge obtained from the preceding tasks
- Each data is trained only once; Available to a flavor of NN, where the typical back propagation is approximate as instantaneous [ref](https://insidebigdata.com/2019/10/29/when-dnns-become-more-human-dnns-vs-transfer-learning-vs-continual-learning/)
- Compute requirements/cost
- Model interpretability
- ==Interview: Understand the algorithm's pro/con with respect to
1. requirements: ie business -> ml objective  
2. constraints: ie on mobile device==

_Different types of Models_
- [TDS: Top10 Algo](https://medium.com/@riteshgupta.ai/10-most-common-machine-learning-algorithms-explained-2023-d7cfe41c2616)
- CNN based architecture (Ch2)
- Text Encoder (Ch4)
- Statistical
- BoW
- TF-IDF
- ML Based
- Embedding Layer
- Word2Vec
- Transformer Based
- Video Encoder (Ch4)
- Video Level Models
- Frame level models
- Multi-task NN (Ch5)
- Matrix Factorization (Ch6)
- Feedback Matrix
- Training Loss
- Squared distance over observed pairs
- Squared distance over all pairs
- weighted combination of observed/un pairs
- Optimization Algo
- SGD
- WALS
- Two Tower Network (Ch6)
- Logistic Regression (LR) (Ch7,8)
- LR + Feature Crossing (Ch 8)
- Decision Tree (Ch7)
- Bagging
- Boosting
- Decision Tree
- GBT
- XGBoost: has L1 and L2 regularization
- Neural Network (Ch7)
- Deep & Cross Network (Ch 8)
- Factorization Machines (Ch 8)
- Deep Factorization Machines

###  4.1 Model Training

##### 4.1.1 Construct Training Data
1. Collect the raw data
   1. Ex: <query, item1:similar, item2:nonSimilar, item3:nonSimilar>
2. Identify Features and Labels
    - labels:
        - human judgments
        - natural
            - implicit:
                - user clicks as proxy for similarity (Ch2)
            - explicit: ratings
3.  Select sampling strategy [ref](https://www.scribbr.com/methodology/sampling-methods/) : select a subset of the population so we get a good representation
    1. stratified sampling:  for each population class, select a subPopulation A that is proportional A's population in the total population
    2. cluster sampling: cluster the population, and select a portion from
4. Split the Data in training, validation, and test [ref](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)
    1. validation is for tuning hyper-parameters
    2. k-fold cross-validation procedure can help
        1. if don't have enough data
        2. for complex computational models (large NN)
        3. This method guarantees that the score of our model does not depend on the way we picked the train and test set.
5. Address class imbalance (if any)
    1. Upsample the minority class; under-sample the majority class
    2. Modify the loss function
        1. class-balanced loss [paper](https://arxiv.org/pdf/1901.05555.pdf)
        2. focal loss [ref1](https://medium.com/swlh/focal-loss-what-why-and-how-df6735f26616) [paper](https://arxiv.org/pdf/1708.02002.pdf)
            1. The focal loss is designed to address the class imbalance by down-weighting the easy examples such that their contribution to the total loss is small even if their number is large
            2. $$CrossEntropy = \sum_{i=0}^{i=classN} y_i \log(p_i)$$
            3. Focal loss adds a $\gamma$ scale factor (<1) to down play the importance of the class we want to pay less attention to$$FocalLoss = \sum_{i=0}^{i=classN} y_i^\gamma \log(p_i)$$

- Book
    - Hand label (Ch5)
    - Adding Booked Listing as Global Context (Ch 9)




##### 4.1.2 Choosing the Loss function (==Training Metric==)
- IMOW: How does MLE/MAP fit into loss functions and optimization algorithm?
    - ==MLE and MAP provides the theoretical framework to derive the loss function, by which optimization algorithm can be applied upon.==

_Maximum Likelihood Estimation (MLE) Vs Maximum Posterior(MAP) Vs Expectation-Maximization Algorithm Vs Least Squared Error and Cross Entropy_

- Terminologies
  - Likelihood :
  - L($\mu\sigma$; data)
  - the likelihood of parameters $\mu\sigma$ given we observed the data
  - Probability:
  - P(data; $\mu\sigma$)
  - probability density of observing the data with model parameters $\mu\sigma$

- Both MLE and MP describe how we learn the parameters of our model to learn the likelihood function L($\mu\sigma$; data)
    - MLE and MAP provide a mechanism to ==derive== the loss function.
        - With alot of math, we see parameters $\theta$ which maximize the log likelihood is to minimize the mean squared error  [tds](https://towardsdatascience.com/estimators-loss-functions-optimizers-core-of-ml-algorithms-d603f6b0161a)
        - We then apply gradient descent on the mean squared error

- Goal of MLE is to infer parameter $\theta$ to infer the most likely likelihood function for our given probability distribution p (Bernouli, Normal, etc..).

$$\theta_{MLE} = {\operatorname{argmax}}  P(X|\theta)$$
$$= {\operatorname{argmax}}  \prod p(x_i|\theta)$$
$$= {\operatorname{argmax}} \log(  \prod p(x_i|\theta) )$$
$$= {\operatorname{argmax}} \sum_{i} \log p(x_i|\theta) )$$

-  Similarly, Maximum Posterior learns the likelihood L($\mu\sigma$; data), but approach from Bayes Theorem
   $$p(\theta|X) = \frac{p(X|\theta) p(\theta)}{p(X)}$$ $$ p(X|\theta) p(\theta)$$
   $$= {\operatorname{argmax}} \sum_{i} \log p(x_i|\theta) + p(\theta)) )$$
   - MAP is similar to MLE but has a prior term p($\theta$)

- In practice, while the log makes the function easier to differentiate to find the minimum, the function can be still intractable.  Expectation maximization algorithm is an ==iterative== algorithm  to estimate parameters.

- Least Squared Error vs Cross Entropy Loss
    - Least Squared error and Cross entropy loss are ways to quantify the difference between between prediction and actual for  scalars and probability distribution, respectively
    - Cross Entropy
      $$H(P,Q) = \sum_{c=1}P(x) \log Q(x)$$
      - Q = probability of prediction
      - P = probability of tree class (true class)
      - c is for a class
- Entropy vs Cross Entropy vs KL Divergence [TDS](https://towardsdatascience.com/entropy-cross-entropy-and-kl-divergence-explained-b09cdae917a)
    - Entropy
        - How much information is gained; how much uncertainty was reduced (more bits more uncertainty reduced)
        - $$H(P,P) = -\sum_{c=1}P(x) \log P(x)$$
        - Example
            - If every day, there is a 35% sunshine of rain and 1% of rain , what is the entropy for 3 days? (how many bits of information)?
              $$Entropy = 3 (-0.35\log(0.35) - 0.01\log(0.01))$$
              $$Entropy = 2.33 bits$$
    - Cross Entropy: takes into account actual vs predicted probability $$H(P,Q) = -\sum_{c=1}P(x) \log Q(x)$$ 
    - KL Divergence (aka relative entropy)
      $$KLDivergence = Entropy - CrossEntropy$$



_SIDE: Training Metric vs Evaluation Metric_
- Training metric is used to learn the parameters (ie cross entropy, square loss)
- often has good mathematical computation behavior
- computationally efficient (log ==> multiplication becomes addition)
- numerically stable (underflow and overflow)
- Evaluation metric (ROC)
- closer to the ML objective
-   ==Summary of Problem Type, Last Layer, and Loss Function==

| Problem Type    | Last Layer Activation | Loss Function  |
| :---        |    :----:   |          ---: |         ---: |
| Classification: Binary      | Sigmoid       | Binary Cross Entropy  |
| Classification: Multi-class, single label    | Softmax        | Categorical Cross Entropy      |
| Classification: Multi-class, Multi-label   | Sigmoid        | Binary Cross Entropy      |
| Regression To Arbitrary Values   | None        | MSE      |
| Regression To Values Between 0 - 1   | Sigmoid        | MSE or Binary Cross Entropy      |


- Cross entropy (Ch5)
  $$BinaryCrossEntropy =  y_i \log(p(i=1)) + (1-y_i)\log(p(i=0))$$
  $$CrossEntropy = \sum_{i=0}^{i=classN} y_i \log(p_i)$$
- Contrastive/Triplet Loss Loss [lilian blog](https://lilianweng.github.io/posts/2021-05-31-contrastive/) (Ch2 Visual Representation Learning)
  - Contrastive loss takes a pair of inputs $(x_i, x_j)$ and learns a function $f_\theta(.)$ that encodes $x_i$ into an embedding vector such that minimizes the embedding distance when they are from the same class but maximizes the distance otherwise. $y_i \in {(1,..., L)}$ is a label among L class
  $$L_{contrastive}(x_i, x_j, \theta)= (y_i=y_j)(f_\theta(x_i)-f_{\theta}(x_j))^2 + (y_i!=y_j)\max(0, \epsilon-(f_\theta(x_i)-f_{\theta}(x_j))^2)$$
  - Triplet loss takes $(anchor, x^+, x^-)$, and learns a function to minimize the distance between anchor and $x^+$ and maximize the distance between anchor and $x^-$
  $$L_{tripliet}=\sum_{x\in X} \max(0, [f(x_{anchor})-f(x^+)]^2 - [f(x_{anchor})-f(x^-)]^2 + \epsilon)$$
    - Huber Loss
    - Residual Squared, MSE, MAE
      $$ residual_{squared} = (y_{predict} - y_{actual})^2$$

$$MAE= \sum_{i=0}^{i=totalPopulation} (y_{actual} - y_{predict}) $$$$MSE= \sum_{i=0}^{i=totalPopulation} (y_{actual} - y_{predict})^2 $$



##### 4.1.3 Regularization
- Context: Bias Variance Tradeoff
    - Bias quantifies how well/close a model's prediction captures the ==true== relationship between system's input and output
        - Models with **high bias** are simple and fail to capture the complexity of the data. Hence, such models lead to a higher training and testing error. 
		 - **Low bias** corresponds to a good fit to the training dataset. Generally, more flexible models result in lower bias.
    - Variance quantifies how well the model ==generalizes== to data set beyond the training dataset
        - It is the amount by which the estimate of the true relationship would change on using a different training dataset.
    - Perspective: Math: Using regression as an example
        - 𝔼|MSE| = 𝔼|$\sum_{i=0}^{i=totalPopulation} (y_{actual} - y_{predict})^2$|
            - 𝔼 is expected value; in this case, it the mean of  MSE of the data set
        - The expected value can be broken down into
            - 𝔼|MSE| = $bias(y_{predict})^2 + var(y_{predict}) + \sigma^2$
                - $\sigma$ is the irreducible error
            - ==This is the bias variance tradeoff mathematically!!==
                - Choose the right model complexity to minimize the expected error, by choosing the right parameter to minimal bias and variance
        - ref: [TDS](https://medium.com/snu-ai/the-bias-variance-trade-off-a-mathematical-view-14ff9dfe5a3c#:~:text=Bias%20is%20defined%20as%20the,%5Bf%CC%82(x)%5D.)

    - Perspective: Bulls Eye [TDS](https://towardsdatascience.com/the-bias-variance-tradeoff-cf18d3ec54f9)
        - low bias; low variance: idea
        - low bias; hi variance: shots centered on bulls eye, but shots are spread apart
        - hi bias; low variance: shots are close to each other, but far from bulls eye
        - hi bias, high variance: shots are from each other and far from the bulls eye

    - Perspective: Model Complexity
        - Complex models (ie large weights, complex architecture NN) can capture the relationship between input and output during training time, but does not generalize well too future unseen data.
            - larger model (num param, large values) can overfit
            - Overfit --> start learning the noise, and won't generalize well.
        - Regularization penalizes complex model so we can improve generalization
- L1 L2 [ref1](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) [ref2](https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning)
    - L1 (aka Lasso Regression):
        - penalize weights in proportion to the sum of ==absolute== value of weight.
        - L1 drive the weights of irrelevant features to 0.
          $$J = \sum_{i=0}^{numSamples}(y_i-\sum_{j=0}^{j=numParams}x_{ij}\beta_j)^2 + \lambda\sum_{j=0}^{j=numParams}|\beta_j|$$
          $$J = Cost Function$$
          $$\beta = Feature Coefficient $$
          $$\lambda = Regularization Parameter$$

    - L2 (aka Ridge Regression):
        - penalize weights in proportion to the sum of value of weight.
        - L2 drives the ==outlier== weights to smaller value.
          $$J = \sum_{i=0}^{numSamples}(y_i-\sum_{j=0}^{j=numParams}x_{ij}\beta_j)^2 + \lambda\sum_{j=0}^{j=numParams}\beta^2_j$$
    - L1 vs L2
        - ==L1 (Lasso) works as feature selection because it shrinks the less important feature's coefficient to 0; yielding a more sparse model==
            - why? [TDS](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
                - L2 ridge regression coefficients have the smallest RSS(loss function) for all points that lie within the circle given by β1² + β2² ≤ s_ ; constraint is a circle
                -  L1 lasso coefficients have the smallest RSS(loss function) for all points that lie within the diamond given by |β1|+|β2|≤ s; constraint  is a diamond
                - Visualize:
                    - for 2 parameter; axises are the coefficient values
                    - L1's constraint of diamond has edges on the axis where the coefficient_j is 0 --> remove features
                - the lasso and ridge regression coefficient estimates are given by the ==first point== at which an ellipse contacts the constraint region
        - L2 (Ridge) is better for model interpretation because it will reduce the coefficient magnitude, but not 0.  We will be able to use the coefficient (aka predictor) to understand what factors are not important.
        - Both reduces overfit increase underfit because larger cofficient value or more parameter will give a bigger cost function.

    - Entropy Regularization [paper](https://paperswithcode.com/method/entropy-regularization)
    - Dropout

##### 4.1.4 Optimization (TODO)
- Once we have framed the loss function, how do we differentiate and update the parameters?

- Common Optimization Methods [deep overview ref](https://www.ruder.io/optimizing-gradient-descent/)
    - SGD
    - AdaGrad
    - Momentum
    - RMSProp


##### 4.1.5 Other Talking Points
- Training from scratch vs fine tuning
- Distributed training
- Negative Sampling (Ch 9, ==p222==)





-----------------------------------------------
# 5 Evaluation
-----------------------------------------------
### Offline Metrics
- Mean Reciprocal Rank (MRR) $$MRR = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{rank_i}$$
    - m = # of lists
    - rank is the position of the ==first== relevant item
    - Con: only consider the first relevant item
- Recall@k $$Recall = \frac{numReleveantItemsInTopkPositions}{totalRelevantItems}$$
    - Con: for some systems, it is hard to know all the relevant items?

- Precision@k$$Precision = \frac{numReleveantItemsInTopkPositions}{k}$$
    - Con:
        - Relevance might need more granularity (ie level of relevance)
        - Applies to only 1 list

- Mean Average Precision: mAP
    - Averages the precision over ==multiple== lists
      $$AP = \frac{\sum_{i=1}^{k}Precision@iIfItemIisRelevant}{numRelevItemInList}$$

- nDCG
    - DCG: cumulative gain of item $$DCG_p=\sum_{i=1}^{p}\frac{rel_i}{log_2(i+1)}$$
        - p = position
        - rel_i  is more granular, ie (1, 2,  3, 4, 5)
    - Normalized DCG (nDCG) $$nDCP_p = \frac{DCG_p}{IDCG_p}$$
        - IDCG_P is when the output of a ==perfect== ranking system

- Confusion matrix
    - Used in classification
    - can be extended to multiclass; 3 classes becomes 3x3 table [ref](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/#Confusion_Matrix_for_Multi-Class_Classification)

|     | Pos Prediction | Neg Prediction  | Metric |
| :---        |    :----:   |          ---: |         ---: |
| Pos Class      | TP       | FN   | _Recall TPR_ |
| Neg Class   | FP        | TN      | _FPR_ |
| Metric   | _Precision_        |       |  |


$$Precision= \frac{TP}{TP + FP}$$
$$Recall=TPR=\frac{TP}{TP + FN}$$
$$FPR = \frac{FP}{FP + TN}$$

	- IMOW:
		- Precision: Of the positive predictions, how many are truly really true?
		- Recall & TPR: Of the positive class, what fraction are we prediction true?
		- FPR: Of the negative class, what fraction are we predicting true? 
		- note: positive class is referred as minority class

- F1 Score
    - Combines recall and precision into 1 number
      $$F1 = \frac{2}{\frac{1}{recall}+\frac{1}{precision}}$$
- ROC curve & ROC-AUC (Ch5)
    - Summarize performance of binary classifier for ==positive== class
    - Plot as the fraction of correct predictions for the positive class (y-axis) versus the fraction of errors for the negative class (x-axis).
        - x-axis: FPR y-axis: TPR
    - Ideal: (x=0, y=1)
        - TPR = 1; we get all the positive class right
        - FPR=0; we have no false positives
    - ==ROC analysis does not have any bias toward models that perform well on the minority class at the expense of the majority class==
    - One point on ROC curve corresponds to 1 threshold.  ROC-AUC averages the ROC across ==all classifier== thresholds, giving us ==one== number we can compare across multiple classifiers.

- PR Curve  and PR-AUC (Ch5)
    - Plots Recall(x) vs Precision(y)
        - Perfect skill is upper right (x-axis=recall=1,  y-axis=precision=1)
    - Precision recall is recommended for high skewed/imbalanced datasets. Using ROC on balanced dataset can yield over-optimistic performance.

    - ROC vs PR AUC [ref](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/)

- Diversity
- Bleu metrics [wiki] : (bilingual evaluation understudy)
    - Typically used in machine translation
- ROGUE []
    - The Recall-Oriented Understudy for Gisting Evaluation (ROUGE) scoring algorithm [1] **calculates the similarity between a candidate document and a collection of reference documents**.

### Online Metrics
- CTR
- Average Time Spent On X
- Video completion Rate
- Total watch time
- Explicit User Feedback
- Conversion Rate
- Revenue Lift
- Bookmark Rate

-----------------------------------------------
# 6 Serving
-----------------------------------------------
- Cloud vs On Device Deployment
- Model Compression [paper](https://arxiv.org/pdf/1710.09282.pdf)
    - Knowledge distillation - Teacher Student model
    - Model pruning - find the least useful parameter and set them to 0. Leads to sparer models
        - General: 5 Types
        1. **Random*:** Simply prune random parameters.
        2. **Magnitude*:** Prune the parameters with the least weight (e.g. their L2 norm).
        3. **Gradient:** Prune parameters based on the accumulated gradient (requires a backward pass and therefore data).
        4. **Information:** Leverage other information such as high-order curvature information for pruning.
        5. **Learned:** Of course, we can also train our network to prune itself (very expensive, requires training)!
    - Pytorch: [blog for pytorch api](https://towardsdatascience.com/how-to-prune-neural-networks-with-pytorch-ebef60316b91) supports random and magnitude
        - Types
            - (local --> per layer) vs (global --> multiple layers)
            - Structured vs Unstructured
                - Unstructured Pruning refers to pruning individual atoms of parameters. E.g. individual weights in linear layers, individual filter pixels in convolution layers, some scaling floats in your custom layer, etc. The point is you prune parameters without their respective structure
                - Structured pruning removes entire structures of parameters. This does not mean that it has to be an entire parameter, but you go beyond removing individual atoms e.g. in linear weights you’d drop entire rows or columns, or, in convolution layers entire filters

|     | Structured | Unstructured  |
| :---        |    :----:   |          ---: |
| Global      | --       | magnitude(==only== L-1 or custom)   |
| Local   | random, magniteude(==any== L-norm or custom)        |  random, magnitude(==only== L-1 or custom)      |


- Quantization - store the parameter with less bits.  Can be done during or post training.
- Testing in Production
    - Shadow deployment: 0% of user see new system
    - A/B testing
- Prediction Pipeline
    - Batch
        - pro: good for processing large amount of data
        - cons:
            - we need to know in advance what is to be generated
            - results will not be _near_ real time
    - Online



### Prediction Pipeline
- Embedding Service (Ch2)
- Nearest Neighbor Service (Ch2)
- Re-ranking Service (Ch2,4)
- Visual Search (Ch4)
- Text Search (Ch4)
- Candidate Generation (Ch 8)
- Ranking (Ch 8)
- Re-Ranking (Ch 8)
- Triton

### Indexing Pipeline
- Indexing Service

### Nearest Neighbor Performance
- Exact match
- Approximate
    - Tree based
    - LSH based
    - Clustering based


-----------------------------------------------
# 7 Monitoring
-----------------------------------------------
- Why system fails in Production
    - Data distribution changes from production
- What to Monitor
    - Operation-related (Graphana)
        - Latency
        - Throughput
        - CPU
        - BW
    - ML
        - monitor input
        - drifts
        - model accuracy
        - model version
- Alarms
    - alarms when metric breaches a threshold?
- OTHER CONSIDERATIONS
    - Cost
        - DynamoDB on AWS has 2 price scheme: on demand or throttled. latter is an order cheaper.
    - Infra needed
    - Integration Points
        - How will model integrate with upstream/downstream data/service/model
    - Data Privacy
        - Ex:
            - Sending PII data to ChatGPT?
