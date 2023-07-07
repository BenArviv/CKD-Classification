# CKD-Classification
This is an assignment given in The Open University of Israel's course: **Data Mining**, in which I was asked to define the problem, preproccess the data, and solve the problem using classification algorithms.

## Defining The Problem
The purpose of the data mining process regarding CKDs is to reveal patterns, dependencies or hidden meanings in the following set of features, describing people's health conditions.
Using statistic tools, algorithms and models, we can check the influence of the given 24 attributes on the possibility of being diagnosed with CKD.
This proccess will help us understand the factors affecting kidney disease, diagnose it earlier, and match the best treatment method or even prevent the disease in the future. 

## Defining the attributes
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

### Defining data mining goals
Prediction of the presence of CKD, depending on different medical indices of a patient.

### Data collection and storage
The data is taken from a single dataset from Bangladesh, and therefore there is no need in integration from different sources.

### Data cleaning
The data contains missing values, so we want to fix the dataset by deleting observations ot filling those values with close values, or by smoothing.
In this case, I chose to fill in missing values using _mean_ (for continous values) and _mode_ (for discrete values).

### Data reduction and transformation
Analyzing if there is a need for data reduction, by deleting observation or irrelevant features, as well as discretizing and normalizing the data.
Notice that some values greatly exceeds the standard deviation, so we fix them.

### Choosing methods and tools for data mining
I've used `Excel` for data organising, as well as `Python` for data proccessing and ML algorithms.

### Matching the data for chosen data mining methods
In order for the data to fit for the different classification algorithms, we will normalize the values such that no single value will have greater weight on the classification.

### Actual data mining proccess
Run the chosen classification algorithms on the optimized data, using `Python`.

### Analyzing the results
Examination of the findings obtained by the classification algorithms, and evaluating them by measures such as degree of accuracy, relevance, simplicity etc., using statistical data analysis.
After examination, there can be two options: <br>
1. **The results are satisfactory** <br>
  In this case, we move forward to the next step - conclusions.<br>
2. **The results aren't satisfactory** <br>
  In this case, we repeat the process while changing some parts of it (e.g data deletion, discretization, different transformations, different algorithms etc.). This action will be repeated until a model is found that satisfies the needs of the test, or until it is decided to stop the process.

### Conclusions
Using the optimal model found to predict the presence of CKD in future patients. Depending on the chosen model, we can present it visually using a mathematical formula, an inference rule, a decision tree or the source code of the algorithm.

### A comparative review between the possible alternatives for performing data mining

#### Logistic Regression
A statistical model used for prediction of a categorical class out of categorical or numerical features, that returns the probability of belonging to a specific class. <br>
**Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm does not require any prior assumptions about the distribution of the random variables in the feature space, and can handle both numerical and categorical values. <br>
**Disadvantages** - the model tends to overfit/underfit if the number of examples is big/small compared to the number if features. In addition, the model is not recommended when the features are co-linear.

#### Information Gain based Decision Tree (ID3)
A supervised learning model for classification tasks, which recuresivly split the data according to the features, with respect to the _information gain_ measure.
**Constraints** - a discretization task must be done for each continuous feature.
**Advantages** - easy to implement and use, and conveniently presentable. Moreover, the algorithm can handle noisy data and missing values, by pruning the tree.
**Disadvantages** - the information gain measure tends to prefer features with larger range of values. In addition, the algorithm might be very expensive, computationally speaking.

#### Gini Index based Decision Tree (CART)
An algorithm that creates a binary decision tree, which recursively splits the data in binary form according to the features, using the _Gini index_ measure, such that after each division, two subgroups are obtained which are as equal in size as possible.
**Advantages** - can handle both numerical and categorical values, as well as missing values. In addition, the obtained tree will more compact, in most cases, compared to other decision tree algorithms.
**Disadvantages** - the Gini index measure tends to prefer features with larger range of values, and the running time is longer, compared to other decision tree algorithms.

#### Random Forest
A classifier that combines the results of many decision trees in order to recieve optimal results.
**Advantages** - can handle high dimensional, large data sets, with missing values. It can also reduce the variance of the data, and thus avoid overfitting, by averaging the predictions obtained by the different trees it creates.
**Disadvantages** - a slow algorithm, mainly on large data sets, and it tends to prefer features with higher number of degrees.
Moreover, this classifier isn't as conveniently presentable as a single decision tree.

### Describe The Stages of Data Preperation

#### Handling missing values
Since there are 24 features in the data set, if for some example more than third of all of the features (8) is missing, this example will be deleted.
Otherwise, fill the missing values with the **mean** (for numerical values) or with the **mode** (for categorical values).

#### Data convertions
Convert all of the categorical values to nominal variables according to their value (e.g _yes_ will recieve the value _1_ etc.), and convert those values to be of type `float` (real numbers). <br>
Apply transofmations according to the chosen classification algorithm: <br>
For the logistic regression algorithm, normalize the nominal data using `min-max` method, so that the new range will be [0,1]. <br>
However, for the random forest algorithm, discretize the numerical values by splitting the data to intervals with equal frequency, so that each values group will get equal representation. <br>
In addition, we'll assume that the data entry process contained errors such as adding an extra '0', or forgetting decimal point etc., therefore for features with high standard deviation or unusual minimal/maximal values with respect to the mean, we'll fix their values manually. 

## Classification

### Choose two methods for classifing the data
The chosen methods are `Logistic Regression` and `Random Forest`.

#### Logistic Regression
A classification algorithm that predicts the probabilities of belonging to some prediction class, in our case - existence of chronic kidney disease. <br>
The algorithm analyzes the connections between one or more independent variables (we'll assume that the features are indeed independent). <br>
The model uses a logistic function to analyze the conditional probabilities between the features, which means it calculates the probability of belonging to a class given the values of some feature.

#### Random Forest
An ensemble algorithm that combines the results obtained by running many decision trees, and returns an optimal classification based on _majority vote_ out of the results of the decision trees. In most cases, the obtained classification is more accurate than that obtained by a single decision tree.

### Describe the steps of the chosen methods
For both methods I've used _10-fold cross-validation_ as well as _bootstrap_ sampling in order to achieve more accurate predictions. Each method was trained on the same examples, but they've been processed differently for each of the algorithms.

#### Logistic Regression
The algorithm is looking for a set of weights for which the prediction error would be minimal.
After cleaning the data, we'll normalize it using the `Min-max` method in order for all the values to be in the interval [0,1], thus there won't be values that influence more than others.
The model will be trained using iterative methods, such as `Gradient Descent`, in order to find the optimal weights that minimize the in-sample error, that would be defined as the least squares loss (_l2-loss_). In addition, we will use regularization to avoid overfitting.

#### Random Forest
The data will be discretized by binning into intervals with equal frequency, so every value of a feature will get equal representation among the rest of the values. I decided that in every run if the algorithm, 100 decision trees will be created, and each division will rely on at most the square root of the total number of features. The splitting criterion will be based on _Gini index_.

## Bayesian and Observational Learning

### Describe and analyze your choice
The chosen algorithm is Naïve-Bayes, because it is a much simpler algorithm than Bayesian network, as the latter takes in account all of the dependencies between the features, while Naïve-Bayes assumes independency, yet achieving good results in practice. A Bayesian network demands a bigger database, from which it can deduce the dependecies between the features. Since the preprocessed database contains just 378 examples, we'll prefer to use the Naïve-Bayes algorithm, which is capable of handling smaller sized databases. Moreover, since the Bayesian network requires many calculations, by using the Naïve-Bayes we get a simpler algorithm whose train and run time will be much faster.

Nevertheless, Naïve-Bayes performence may suffer if there are correlated feature, and thus we would want to perform dimensionality reduction (e.g by using the `PCA` algorithm), to achieve a data set with lower correlation.

The algorithm calculates the conditional probabilities of each of the features given the classification class, and then, for any new unseen example, calculates the product of the relevant conditional probabilities, given each one of the classes. The class that achieves the highest probability is the final classification of the algorithm.

### Describe and analyze the task using K-NN algorithm
`K-NN` is a classification algorithm which treats the data as points in a space with a dimension the size of the number of features.

This algorithm requires no training, since each example is being calculated independently of the other examples, but the _k_ nearest examples to it, based on some distance function.
