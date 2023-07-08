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

### Actual Data Mining Proccess
Run the chosen classification algorithms on the optimized data, using `Python`.

### Analyzing The Results
Examination of the findings obtained by the classification algorithms, and evaluating them by measurements such as _degree of accuracy_, _relevance_, _simplicity_ etc., using statistical data analysis.
After examination, there can be two options: <br>
1. **The results are satisfactory** <br>
  In this case, we move forward to the next step - conclusions.<br>
2. **The results aren't satisfactory** <br>
  In this case, we repeat the process while changing some parts of it (e.g data deletion, discretization, different transformations, different algorithms etc.). This action will be repeated until a model is found that satisfies the needs of the test, or until it is decided to stop the process.

### Conclusions
Using the optimal model found to predict the presence of CKD in future patients. Depending on the chosen model, we can present it visually using a mathematical formula, an inference rule, a decision tree or the source code of the algorithm.

### A Comparative Review Between The Possible Alternatives For Performing Data Mining

#### Logistic Regression
A statistical model used for prediction of a categorical class out of categorical or numerical features, that returns the probability of belonging to a specific class. <br>
- **Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm does not require any prior assumptions about the distribution of the random variables in the feature space, and can handle both numerical and categorical values. <br>
- **Disadvantages** - the model tends to overfit/underfit if the number of examples is big/small compared to the number of features. In addition, the model is not recommended when the features are co-linear.

#### Information Gain based Decision Tree (ID3)
A supervised learning model for classification tasks, which recuresively splits the data according to the features, with respect to the _information gain_ measure.
- **Constraints** - a discretization task must be done for each continuous feature.
- **Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm can handle noisy data and missing values, by pruning the tree.
- **Disadvantages** - the _information gain_ measure tends to prefer features with larger range of values. In addition, the algorithm might be very expensive, computationally speaking.

#### Gini Index based Decision Tree (CART)
An algorithm that creates a binary decision tree, which recursively splits the data in a binary form according to the features, using the _Gini index_ measure, such that after each division, two subgroups are obtained which are as equal in size as possible.
- **Advantages** - can handle both numerical and categorical values, as well as missing values. In addition, the obtained tree will be more compact, in most cases, compared to other decision tree algorithms.
- **Disadvantages** - the _Gini index_ measure tends to prefer features with larger range of values, and the running time is longer, compared to other decision tree algorithms.

#### Random Forest
A classifier that combines the results of many decision trees in order to achieve optimal results.
- **Advantages** - can handle high dimensional, large data sets, with missing values. It can also reduce the variance of the data, and thus avoid overfitting, by averaging the predictions obtained by the different trees it creates.
- **Disadvantages** - a slow algorithm, mainly on large data sets, and it tends to prefer features with higher number of degrees.
Moreover, this classifier isn't as conveniently presentable as a single decision tree.

### Describe The Stages of Data Preperation

#### Handling Missing Values
Since there are 24 features in the data set, if for some example more than third of all of the features (8) is missing, this example will be deleted.
Otherwise, fill the missing values with the **mean** (for numerical values) or with the **mode** (for categorical values).

#### Data Convertions
Convert all of the categorical values to nominal variables according to their value (e.g _yes_ will recieve the value _1_ etc.), and convert those values to be of type `float` (real numbers). <br>
Apply transofmations according to the chosen classification algorithm: <br>
For the `Logistic Regression` algorithm, normalize the nominal data using `min-max` method, so that the new range will be [0,1]. <br>
However, for the `Random Forest` algorithm, discretize the numerical values by splitting the data to intervals with equal frequency, so that each values group will get equal representation. <br>
In addition, we'll assume that the data entry process contained errors such as adding an extra '0', or forgetting decimal point etc., therefore for features with high standard deviation or unusual minimal/maximal values with respect to the mean, we'll fix their values manually. 

## Classification

### Choose Two Methods For Classifing The Data
The chosen methods are `Logistic Regression` and `Random Forest`.

#### Logistic Regression
A classification algorithm that predicts the probabilities of belonging to some prediction class, in our case - the existence of chronic kidney disease. <br>
The algorithm analyzes the connections between one or more independent variables (we'll assume that the features are indeed independent). <br>
The model uses a logistic function to analyze the conditional probabilities between the features, which means it calculates the probability of belonging to a class given the values of some feature.

#### Random Forest
An ensemble algorithm that combines the results obtained by running many decision trees, and returns an optimal classification based on _majority vote_ out of the results of the decision trees. In most cases, the obtained classification is more accurate than that obtained by a single decision tree.

### Describe The Steps Of The Chosen Methods
For both methods I've used _10-fold cross-validation_ as well as _bootstrap_ sampling in order to achieve more accurate predictions. Each method was trained on the same examples, but they've been processed differently for each of the algorithms.

#### Logistic Regression
The algorithm is looking for a set of weights for which the prediction error would be minimal.
After cleaning the data, we'll normalize it using the `min-max` method in order for all the values to be in the interval [0,1], thus there won't be values that influence more than others.
The model will be trained using iterative methods, such as `Gradient Descent`, in order to find the optimal weights that minimizes the in-sample error, that would be defined as the least squares loss (_l2-loss_). In addition, we will use regularization to avoid overfitting.

#### Random Forest
The data will be discretized by binning into intervals with equal frequency, so every value of a feature will get equal representation among the rest of the values. I decided that in every run of the algorithm, 100 decision trees will be created, and each division will rely on at most the square root of the total number of features. The splitting criterion will be based on _Gini index_.

## Bayesian and Observational Learning

### Describe And Analyze Your Choice
The chosen algorithm is `Naïve-Bayes`, because it is a much simpler algorithm than `Bayesian network`, as the latter takes into account all of the dependencies between the features, while `Naïve-Bayes` assumes independency, yet achieving good results in practice. A `Bayesian network` demands a bigger database, from which it can deduce the dependecies between the features. Since the preprocessed database contains just 378 examples, we'll prefer to use the `Naïve-Bayes` algorithm, which is capable of handling smaller sized databases. Moreover, since the `Bayesian network` requires many calculations, by using the `Naïve-Bayes` we get a simpler algorithm whose train and run time will be much faster.

Nevertheless, `Naïve-Bayes` performence may suffer if there are correlated feature, and thus we would want to perform dimensionality reduction (e.g by using the `PCA` algorithm), to achieve a data set with lower correlation.

The algorithm calculates the conditional probabilities of each of the features given the classification class, and then, for any new unseen example, calculates the product of the relevant conditional probabilities, given each one of the classes. The class that achieves the highest probability is the final classification of the algorithm.

### Describe And Analyze The Task Using K-NN Algorithm
`K-NN` is a classification algorithm that treats the data as points in a space with a dimension the size of the number of features.

This algorithm requires no training, since each example is being calculated independently of the other examples, but the _k_ nearest examples to it, based on some distance function. Therefore the algorithm is simple to implement, and efficient in terms of time. In order to implement it, all that is needed is to calculate the distances between all of the points, and then search the nearest neighbors and classify the new example according to them.

Since it requires those calculations, we can infer that for larger databases or a high number of features, it won't be as effective, because for each new classification it calculates the distance from all of the points. Furthermore, the values of each of the features need to be normalized, so that the algorithm will not be biased in favor of features with higher values.

In order to classify a new example, the algorithm calculates the distance of the new point and all of the other points in the database, and chooses the _k_ closest points. After that, the classification is immediate and carried out by choosing the majority vote between all of the neighbors.

### Choose One Method And Explain Your Choice
The chosen algorithm is `Naïve-Bayes`, because generally, it will be faster to train and classify than `KNN`, since it calculates the probabilities under the independency assumption, unlike KNN that requires saving and calculating the distances of all the examples for the classification task. Furthermore, the independency assumption simplifies the model, since it refers to a smaller set of hypotheses, that contains only those in which the features are independent, unlike `KNN` that doesn't make any simplifying assumptions. In addition, `Naïve-Bayes` algorithm is simpler to interpret than `KNN`, since the decisions that the model makes are based on just conditional probabilities between each feature and class, therefore its is clear how a result is obtained. In contrast, `KNN` requires calculating distance and then classifies according to the majority vote, therefore the classification task is less intuitive to understand.

### Run And Report The Results Of The Algorithm
I used `Python` to run the algorithm.
Similar to the classification performed using the previous algorithms, I applied _k-fold cross-validation_ with _k=10_. In every iteration, we perform _bootstrap_ sampling to achieve the training set, while the rest will be the test set.

## Cluster Analysis

### Define Quality Measurements For Clusters
In order to evaluate the quality of a cluster created by a cluster analysis algorithm, we'll use the next measurements:
- **Distance within the cluster** - we want all the elements in a cluster to be as similar as possible. We calculate the sum of squared distances between each pair of points in the cluster, such that this sum will be minimal.
- **Distance between the clusters** - we want each cluster to be as different as possible compared to other clusters. We do the same calculation as before, aiming the sum to be maximal.
- **Homogenity** - we want each cluster to contain elements that belong to the same class.
- **Completeness** - we want examples that are classified to the same class, to belong to the same cluster.
- **V-measure** - the harmonic mean between the _homogenity_ and the _completeness_, which describes the trade-off between these two.
- **ARI measurement** - checks the similarity between _actual_ and _predicted_ classifications, and the _degree of randomness_ in classifying new examples.
- **AMI measurement** - checks the mutual information between the _actual_ and _predicted_ classifications, similar to _ARI measurement_.
- **Silhouette** - checks the degree to which an example matches the cluster to which it belongs according to the distance from points within the cluster and points in near clusters.

### Choose One Method For Cluster Analysis
The chosen algorithm is `DBScan`.
It is a density-based clustering algorithm that run under the assuumption that clusters are areas with high density, that are separated by areas with lower density. Since it recognizes clusters according to density, rather than distance, it is able to identify clusters with arbitrary shape, which isn't necessarily elliptical. In addition, it makes no assumptions about the distribution of the data.
This is a robust algorithm, and it is not sensitive to outliers or noise, but on the other hand, it is computationally expensive, as it goes through all of the points and calculates distances between them, for every _core point_. Since our database doesn't contain a large amount of observations, we can neglect the computation cost of the algorithm, and in return, achieve more acccurate clusters.

### Describe the cluster analysis steps for the chosen approach
In order to run `DBScan` we need to preprocess the data, in which we'll handle missing values (by completion or deletion), conversion of the values to numerical values (even though the algorithm is capable of handling categorical values), and normalization of the data so that features like `Age` won't get higher importance in the classification task. Also, in order to achieve clearer clusters that are easily interpretable, we'll apply a dimensionality reduction algorith, such as `Principal Component Analysis`. The `PCA` algorithm practically project the data into a lower dimensional vector space, such that the variance is maximal. To do so, the algorithm calculates the _eigenvalues_ of each of the features, and keeps the _n_ largest ones, where _n_ is the wanted dimension.

Since we are working with the optimized database, there is no need in another processing of the data, as they already match our needs, except for the dimensionality reduction.

The `DBScan` algorithm doesn't require the expected number of clusters from the users, however it uses a parameter _ε_ that defines the neighborhood radius of each point, as well as a parameter _MinPts_, which, as its name suggests, is the minimal number of points in the _ε_-neighborhood of a point.

The algorithm divides the points into three groups - **core** points, **border** points and **noise** points, which are characterized by the following features:
A point _p_ is the **core point**, if there are at least _MinPts_ points in her _ε_-neighborhood.
If the distance between the points _p_ and _q_, where _p_ is a core point, is less than _ε_, we say tha _q_ is _directly reachable from p_.
If there is a path between _p_ and _q_, in which each pair of consecutive points are directly reachable, we say that _q_ is _reachable from p_.
Every point that is not reachable is considered as **outlier** or **noise point**.

The algorithm works as follows:
For each point, find all of the points in its _ε_-neighborhood.
If the number of points found is larger than _MinPts_, mark this point as a **core point**.
For each core point, find all the reachable neighbors of it.
Continue this process until there are no more core points to check.

### Run and report the results of the algorithm
I used python to run the algorithm.
The required parameters will be defined as _MinPts_=6, _ε_=0.5, and we apply `PCA` in order to reduce the dimensions, from 24 to 2, for visibility.

## Artificial Neural Network

### Define the network's architecture
We define the neural network in the following way.
The network will contain three layer types:
- **Input layer** - contain 24 neurons, since the data is taken from a 24-dimensional space.
- **Hidden layers** - between the input and the output layers. After a trial and error process, we choose _two_ hidden layers, each containing 10 neurons.
- **Output layer** - last layer in the network, contains two neurons, according to the number of classes in our task.
The netwrok will be defines as _feed-forward_ and _fully-connected_, meaning that the direction of the network is only forward, from the input layer towards the output layer, and each neuron in a layer is connected to all of the neurons in the next layer.
The activation function will be `ReLU`.
The loss function will be `Cross-entropy`, with the optimization done using `Stochastic Gradient Descent`, and defines as: <br>
H_p (q)=-1/N ∑_(n=1)^N▒〖y_n⋅log⁡(p(y_n ))+(1-y_n )⋅log⁡(1-p(y_n )) 〗<br>
where _y_ is the actual class, and _p(y)_ is the probability of predicting the n'th point, out of N points.

### Define the optimization parameters
The optimization parameters are the batch's size, learning rate, regularization parameters and optimization algorithm.
As mentioned, we choose the algorithm to be `SGD`, where each bach will be of size 100. The regularization parameter will be left as _α=0.0001_, and the learning rate will be _l=0.001_, according to the default values in `SKLearn` library.

### Run and report the results of the algorithm
I used `Python` to run the network.
First, we divide the dataset into a training set (80%) and a test set (20%) and perform _k-fold cross-validation_, such that in each fold we divide the training set to a training set (80%) and a validation set (20%), so that we can measure the classifier's accuracy.
