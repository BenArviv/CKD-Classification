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
  In this case, we move forward for the next step - conclusions.<br>
2. **The results aren't satisfactory** <br>
  In this case, we repeat the proccess while changing some parts of it (e.g data deletion, discretization, different transformations, different algorithms etc.). We will continue like this until a model is found that satisfies the needs of the test, or until it is decided to stop the process.

### Conclusions
Using the optimal model found to predict the presence of CKD in future patients. Depending on the chosen model, we can present it visually using a mathematical formula, an inference rule, a decision tree or the source code of the algorithm.
