# CKD-Classification
This is an assignment given in The Open University of Israel's course: **Data Mining**, in which I was asked to define the problem, preproccess the data, and solve the problem using classification algorithms.

## Defining The Problem
The purpose of the data mining process regarding CKDs is to reveal patterns, dependencies or hidden meanings in the following set of features, describing people's health conditions.
Using statistic tools, algorithms and models, we can check the influence of the given 24 attributes on the possibility of being diagnosed with CKD.
This proccess will help us understand the factors affecting kidney disease, diagnose it earlier, and match the best treatment method or even prevent the disease in the future. 

## Defining The Attributes
| # | Attribute name | Description | Data type | Unit measures | Range of values | Average | Standard deviation |
| - | -------------- | ---------- | --------- | -------------- | ------------------ | ------- | ------------------ |
| 1 | age | Age | numeric | years | 2-90 | 51.48338 | 17.16971 |
| 2 | bp | Blood pressure | numeric | Mm/Hg | 50-180 | 76.46907 | 13.68364 |
| 3 | sg | Specific gravity | nominal | | {1.005,1.010,1.015,1.020,1.025} | | |
| 4 | al | Albumin | nominal | | {0,1,2,3,4,5} | | |
| 5 | su | Sugar | nominal | | {0,1,2,3,4,5} | | |
| 6 | rbc | Red blood cells | nominal | | {Normal, Abnormal} | | |
| 7 | pc | Pus cell | nominal | | {Normal, Abnormal} | | |
| 8 | pcc | Puss cell clumps | nominal | | {Present, Not present} | | |
| 9 | ba | Bacteria | nominal | | {Present, Not present} | | |
| 10 | bgr | Blood glucose random | numeric | Mgs/Dl | 22-490 | 148.0365 | 72.28171 |
| 11 | bu | Blood urea | numeric | Mgs/Dl | 1.5-391 | 57.42572 | 50.50301 |
| 12 | sc | Serum creatinine | numeric | Mgs/Dl | 0.4-76 | 3.072454 | 5.741126 |
| 13 | sod | Sodium | numeric |mEq/L | 4.5-163 | 137.5288 | 10.40875 |
| 14 | pot | Potassium | numeric | mEq/L | 2.5-47 | 4.627244 | 3.19304 |
| 15 | hemo | Hemoglobin | numeric | Gms | 3.1-17.8 | 12.52644 | 2.912587 |
| 16 | pcv | Packed cell volume | numeric | | 9-54 | 38.8845 | 8.990105 |
| 17 | wc | White blood cell count | numeric | Cells/cumm | 2200-26400 | 8406.122 | 2944.474 |
|18 | rc | Red blood cell count | numeric | Millions/cmm | 2.1-8 | 4.707435 | 1.025323 |
| 19 | htn | Hypertension | nominal | | {Yes, No} | | |
| 20 | dm | Diabetes mellitus | nominal | | {Yes, No} | | |
| 21 | cad | Coronay artery disease | nominal | | {Yes, No} | | |
| 22 | appet | Appetite | nominam | | {Good, Poor} | | |
| 23 | pe | Pedel edema | nominal | | {Yes, No} | | |
| 24 | ane | Anemia | nominal | | {Yes, No} | | |

## Defining The KDD Steps

### Defining Data Mining Goals
Prediction of the presence of **CKD**, depending on different medical indices of a patient.

### Data Collection And Storage
The data is taken from a single dataset from Bangladesh, and therefore there is no need in integration from different sources.

### Data Cleaning
The data contains missing values, so we want to fix the dataset by deleting observations ot filling those values with close values, or by smoothing.
In this case, I chose to fill in missing values using _mean_ (for continous values) and _mode_ (for discrete values).

### Data Reduction And Transformation
Analyzing if there is a need for data reduction, by deleting observation or irrelevant features, as well as discretizing and normalizing the data.
Notice that some values greatly exceeds the standard deviation, so we fix them.

### Choosing Methods And Tools For Data Mining
I've used `Excel` for organising the data, as well as `Python` for data proccessing and running ML algorithms.

### Matching The Data For Chosen Data Mining Methods
In order for the data to fit for the different classification algorithms, we will normalize the values such that no single value will have greater weight on the classification.

### Actual Data Mining Process
Run the chosen classification algorithms on the optimized data, using `Python`.

### Analyzing The Results
Analyzing the results derived from the classification algorithm, and assessing them based on metrics like _accuracy_, _relevance_, _simplicity_ and so on, through the use of statistical data analysis.
After examination, the results can be either of the following:  <br>
1. **The results are satisfactory** <br>
  In this case, we move forward to the next step - conclusions.<br>
2. **The results aren't satisfactory** <br>
  In this case, we repeat the process while changing some parts of it (e.g data deletion, discretization, different transformations, different algorithms etc.). This action will be repeated until a model is found that satisfies the needs of the test, or until it is decided to stop the process.

### Conclusions
Using the optimal model found to predict the presence of CKD in future patients. Depending on the chosen model, we can present it visually using a mathematical formula, an inference rule, a decision tree or the source code of the algorithm.

### A Comparative Review Between The Possible Alternatives For Performing Data Mining

#### Logistic Regression
A statistical model employed for forecasting a categorical class from either categorical or numerical attributes, which provides the likelihood of association with a particular class. <br>
- **Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm does not require any prior assumptions about the distribution of the random variables in the feature space, and can handle both numerical and categorical values. <br>
- **Disadvantages** - the model tends to overfit/underfit if the number of examples is big/small compared to the number of features. In addition, the model is not recommended when the features are co-linear.

#### Information Gain based Decision Tree (ID3)
A supervised learning model designed for classification tasks, which iteratively divides the data based on the features, in relation to the measure of _information gain_.
- **Constraints** - a discretization task must be done for each continuous feature.
- **Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm can handle noisy data and missing values, by pruning the tree.
- **Disadvantages** - the _information gain_ metric generally favors attributes with a broader spectrum of values. Moreover, the algorithm could be quite demanding in terms of computational resources.

#### Gini Index based Decision Tree (CART)
An algorithm that constructs a binary decision tree, which iteratively divides the data into two based on the features, utilizing the _Gini index_ metric, so that each split results in two subgroups that are as similar in size as feasible.
- **Advantages** - can manage both numerical and categorical data, along with missing values. Furthermore, the resulting tree is typically more concise compared to other decision tree algorithms.
- **Disadvantages** - the _Gini index_ measure tends to prefer features with larger range of values, and the running time is longer, compared to other decision tree algorithms.

#### Random Forest
A classifier that combines the outcomes of numerous decision trees to attain the best possible results.
- **Advantages** - can manage high-dimensional, extensive datasets, including those with missing values. It also has the ability to decrease the data's variance, thereby preventing overfitting, by averaging the predictions generated by the various trees it constructs.
- **Disadvantages** - an algorithm that operates slowly, particularly on extensive datasets, and it has a tendency to favor features with a greater number of degrees. Additionally, this classifier is not as easily representable as a solitary decision tree.
  
### Describe The Stages of Data Preperation

#### Handling Missing Values
Given that the dataset contains 24 features, any example with more than a third (8) of its features missing will be removed. If not, the missing values will be replaced with the **mean** (for numerical values) or the **mode** (for categorical values).

#### Data Transformation
Transform all categorical values into nominal variables based on their value (for instance, 'yes' will be assigned the value $1$, etc.), and convert these values into `float` type (real numbers).
Implement transformations in accordance with the selected classification algorithm: <br>
For the `Logistic Regression` algorithm, normalize the nominal data using `min-max` method, so that the new range will be [0,1]. <br>
However, for the `Random Forest` algorithm, discretize the numerical values by splitting the data to intervals with equal frequency, so that each values group will get equal representation. <br>
Furthermore, we'll consider the possibility of errors in the data entry process, such as an extra '0' added, or a missed decimal point, etc. Hence, for features with a high standard deviation or abnormal minimum/maximum values relative to the mean, we'll manually adjust their values.

## Classification

### Select Two Techniques for Data Classification
The chosen techniques are `Logistic Regression` and `Random Forest`.

#### Logistic Regression
A classification algorithm that estimates the likelihoods of association with a certain prediction class, in our scenario - the presence of chronic kidney disease. The algorithm examines the relationships between one or more independent variables (we'll operate under the assumption that the features are indeed independent). The model employs a logistic function to scrutinize the conditional probabilities among the features, implying it computes the probability of class membership given the values of a particular feature.

#### Random Forest
An ensemble algorithm that combines the outcomes achieved by executing numerous decision trees, and delivers an optimal classification based on the majority vote from the results of the decision trees. In the majority of instances, the classification obtained is more precise than that derived from a single decision tree.

### Describe The Steps Of The Chosen Methods
For both techniques, I employed _10-fold cross-validation_ and _bootstrap sampling_ to obtain more precise predictions. Each method was trained using the same examples, but the examples were processed differently for each algorithm.

#### Logistic Regression
The algorithm seeks a set of weights that would minimize prediction error. After data cleansing, we'll normalize it using the `min-max` method to ensure all values fall within the $[0,1]$ range, preventing any value from having an undue influence. The model will be trained using iterative methods, like `Gradient Descent`, to find the optimal weights that minimize the in-sample error, defined as the least squares loss (_l2-loss_). Additionally, we'll employ regularization to prevent overfitting.

#### Random Forest
The data will be discretized by binning into intervals with equal frequency, so every value of a feature will get equal representation among the rest of the values. I decided that in every run of the algorithm, 100 decision trees will be created, and each division will rely on at most the square root of the total number of features. The splitting criterion will be based on _Gini index_.

## Bayesian and Observational Learning

### Describe And Analyze Your Choice
The selected algorithm is `Naïve-Bayes` due to its relative simplicity when compared to the `Bayesian network`. The latter accounts for all feature interdependencies, while `Naïve-Bayes` operates under the assumption of feature independence, yet still delivers commendable results in practice. A `Bayesian network` requires a larger database to infer feature dependencies. Given that the preprocessed database only contains 378 examples, the `Naïve-Bayes` algorithm, which can handle smaller databases, is the preferred choice. Furthermore, as the `Bayesian network` necessitates numerous computations, opting for `Naïve-Bayes` provides a simpler algorithm with significantly faster training and execution times.

However, `Naïve-Bayes` performance could be compromised if there are correlated features, prompting us to carry out dimensionality reduction (for instance, by using the `PCA` algorithm) to obtain a dataset with reduced correlation.

The algorithm computes the conditional probabilities of each feature given the classification class, and then, for any new unseen example, it calculates the product of the relevant conditional probabilities for each class. The class that yields the highest probability becomes the final classification of the algorithm.

### Describe And Analyze The Task Using K-NN Algorithm
`K-NN` is a classification algorithm that views the data as points within a space, the dimension of which equals the number of features.

This algorithm doesn't necessitate training as each example is computed independently of others, but in relation to its $k$ nearest examples, based on a certain distance function. Hence, the algorithm is straightforward to implement and time-efficient. To execute it, all that's required is to compute the distances between all points, then identify the nearest neighbors and classify the new example based on them.

Given that it necessitates these calculations, we can deduce that for larger databases or a high number of features, it won't be as effective, as it calculates the distance from all points for each new classification. Additionally, the values of each feature need to be normalized to prevent the algorithm from favoring features with higher values.

To classify a new example, the algorithm computes the distance between the new point and all other points in the database, selecting the $k$ closest points. Following this, the classification is immediate and is determined by the majority vote among all the neighbors.

### Choose One Method And Explain Your Choice
The chosen algorithm is `Naïve-Bayes`, primarily because it's generally faster to train and classify than `KNN`. This is because it calculates probabilities under the assumption of independence, unlike `KNN` which necessitates storing and computing the distances of all examples for the classification task. Moreover, the assumption of independence simplifies the model as it pertains to a smaller set of hypotheses, only those where the features are independent, unlike `KNN` which doesn't make any simplifying assumptions. Additionally, the `Naïve-Bayes` algorithm is easier to interpret than `KNN` since the model's decisions are based solely on conditional probabilities between each feature and class, making it clear how a result is obtained. Conversely, `KNN` requires calculating distance and then classifies based on majority vote, making the classification task less intuitive to understand.

### Run And Report The Results Of The Algorithm
I utilized `Python` to execute the algorithm. Analogous to the classification conducted using the previous algorithms, I implemented _k-fold cross-validation_ with $k=10$. In each iteration, we carry out _bootstrap sampling_ to obtain the training set, while the remainder forms the test set.

## Cluster Analysis

### Define Quality Measures for Clustering
In order to evaluate the quality of a cluster created by a cluster analysis algorithm, we'll use the next measurements:
- **Distance within the cluster** - we aim for all elements within a cluster to be as similar as possible. This is achieved by calculating the sum of squared distances between each pair of points within the cluster, with the goal of minimizing this sum.
- **Distance between the clusters** - we aim for each cluster to be as distinct as possible from other clusters. We perform the same calculation as before, but this time we aim for the sum to be maximal.
- **Homogenity** - we desire each cluster to contain elements that belong to the same class.
- **Completeness** - we want examples that are classified to the same class to belong to the same cluster.
- **V-measure** - the harmonic mean between _homogeneity_ and _completeness_, which describes the trade-off between these two.
- **ARI measurement** - assesses the similarity between _actual_ and _predicted_ classifications, and the _degree of randomness_ in classifying new examples.
- **AMI measurement** - evaluates the mutual information between the _actual_ and _predicted_ classifications, similar to _ARI measurement_.
- **Silhouette** - examines the degree to which an example matches the cluster to which it belongs based on the distance from points within the cluster and points in nearby clusters.

### Choose One Method For Cluster Analysis
The selected algorithm is `DBScan`. This is a density-based clustering algorithm that operates under the assumption that clusters are high-density areas separated by lower-density areas. As it identifies clusters based on density rather than distance, it can detect clusters of arbitrary shapes, not necessarily elliptical. Furthermore, it doesn't make assumptions about the data distribution. This robust algorithm is not sensitive to outliers or noise. However, it is computationally expensive as it iterates through all points and calculates distances between them for every _core point_. Given that our database doesn't contain a large number of observations, we can overlook the computational cost of the algorithm and, in return, achieve more accurate clusters.

### Describe the cluster analysis steps for the chosen approach
To run `DBScan`, we need to preprocess the data. This involves handling missing values (either by completion or deletion), converting values to numerical values (even though the algorithm can handle categorical values), and normalizing the data so that features like _Age_ don't have undue importance in the classification task. To achieve clearer, easily interpretable clusters, we'll apply a dimensionality reduction algorithm, such as `Principal Component Analysis` (PCA). The `PCA` algorithm essentially projects the data into a lower-dimensional vector space to maximize variance. It does this by calculating the _eigenvalues_ of each feature and retaining the $n$ largest ones, where $n$ is the desired dimension.

Given that we are working with the optimized database, there's no need for further data processing as the data already meet our requirements, with the exception of dimensionality reduction.

The `DBScan` algorithm doesn't necessitate the user to provide the expected number of clusters. However, it does use a parameter $\varepsilon$ that defines the neighborhood radius of each point, and a parameter _MinPts_, which, as the name implies, is the minimum number of points required in the $\varepsilon$-neighborhood of a point.

The algorithm categorizes points into three groups - **core** points, **border** points, and **noise** points, characterized by the following features:

- A point $p$ is a **core point** if there are at least _MinPts_ points within its $\varepsilon$-neighborhood.
- If the distance between points $p$ and $q$, where $p$ is a core point, is less than $\varepsilon$, we say that $q$ is _directly reachable_ from $p$.
- If there's a path between $p$ and $q$ where each pair of consecutive points are directly reachable, we say that $q$ is _reachable_ from $p$.
- Any point that isn't reachable is considered an **outlier** or **noise point**.

The algorithm operates as follows:

1. For each point, identify all points within its $\varepsilon$-neighborhood.
2. If the number of points found exceeds _MinPts_, mark this point as a **core point**.
3. For each core point, find all its reachable neighbors.
4. Continue this process until there are no more core points to check.

### Run and report the results of the algorithm
I utilized Python to execute the algorithm. The necessary parameters are defined as _MinPts_=6, $\varepsilon$=0.5, and we apply `PCA` to reduce the dimensions from 24 to 2 for visibility purposes.

## Artificial Neural Networks

### Define the network's architecture
The neural network is defined as follows. The network will consist of three types of layers:

- **Input layer** - contains 24 neurons, corresponding to the data being taken from a 24-dimensional space.
- **Hidden layers** - these are situated between the input and output layers. After a process of trial and error, we choose _two_ hidden layers, each containing 10 neurons.
- **Output layer** - this is the final layer in the network and contains two neurons, corresponding to the number of classes in our task.

The network is defined as a _feed-forward_, _fully-connected_ network, meaning that the direction of the network is strictly forward, from the input layer to the output layer, and each neuron in a layer is connected to all neurons in the subsequent layer.

The activation function will be `ReLU`.

The loss function will be `Cross-entropy`, with optimization performed using `Stochastic Gradient Descent`. The loss function is defined as follows:<br>
$H_p (q) = -1/N \cdot \sum_{n=1}^N (y_n \cdot log (p(y_n)) + (1-y_n) \cdot log (1-p(y_n)))$<br>
where _y_ is the actual class, and _p(y)_ is the probability of predicting the n'th point, out of N points.

### Define the optimization parameters
The optimization parameters are the batch's size, learning rate, regularization parameters and optimization algorithm.
As mentioned, we choose the algorithm to be `SGD`, where each bach will be of size 100. The regularization parameter will be left as _α=0.0001_, and the learning rate will be _l=0.001_, according to the default values in `SKLearn` library.

### Run and report the results of the algorithm
I used `Python` to run the network.
First, we divide the dataset into a training set (80%) and a test set (20%) and perform _k-fold cross-validation_, such that in each fold we divide the training set to a training set (80%) and a validation set (20%), so that we can measure the classifier's accuracy.
