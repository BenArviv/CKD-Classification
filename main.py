import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from scipy.io.arff import loadarff
import random
import math

def bootstrap_sample(X, Y, n):
    X_res = []
    Y_res = []
    length = len(X)
    indices = np.zeros(length)
    index = -1
    
    for i in range(n):
        index = random.randint(0, length - 1)
        X_res.append(X[index])
        Y_res.append(Y[index])
        indices[index] = 1
        
    return X_res, Y_res, indices

def normalize_transform(df, cls): # performs data normalization
    "data normalization"
    for col in ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']: # numerical data
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) # min-max normalization

    X = df.drop(cls, axis = 1)
    X = X.to_numpy()
    Y = df[cls]
    Y = Y.to_numpy()
    
    return X, Y

def discretize_transform(df, cls): # performs data discretization 
    "data discretization"
    for col in ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
        average = df[col].mean()
        ds = df[col].std()
        factor = int(average / (2 * ds) + 1)
        df[col] = pd.qcut(df[col], factor, labels = False, duplicates = 'drop') # equal frequency discretization
        
    X = df.drop(cls, axis = 1)
    X = X.to_numpy()
    Y = df[cls]
    Y = Y.to_numpy()
    
    return X, Y

def detailed_accuracy(Y_test, Y_pred, TPR, FPR, auc): # detailed accuracy
    df = pd.DataFrame(columns=['TP', 'FP', 'Precision', 'Recall', 'F-Measure', 'AUC', 'Class', 'Accuracy', 'Error Rate'])
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for i in range(len(Y_pred)):
        if Y_test[i] == 0 and Y_pred[i] == 0:
            TN += 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            FN += 1
        elif Y_test[i] == 1 and Y_pred[i] == 0:
            FP += 1
        elif Y_test[i] == 1 and Y_pred[i] == 1:
            TP += 1
            
    Precision = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    F_Measure = 2 * Precision * Recall / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Error_Rate = (FP + FN) / (TP + TN + FP + FN)
    
    df.loc[1] = [TP, FP, Precision, Recall, F_Measure, auc, 'ckd', '', '']
    df.loc[2] = [TN, FN, NPV, Specificity, F_Measure, auc, 'notckd', '', '']
    df.loc[3] = ['', '', '', '', '', '', 'Total:', Accuracy, Error_Rate]
    
    return df

def mean_squared_error(Y_test, Y_pred): # mean squared error
    length = len(Y_test)
    sum = 0
    
    for i in range(length):
        sum += (Y_test[i] - Y_pred[i]) ** 2
    
    return sum / length

def logistic_regression_model(X, Y, k): # logistic regression model
    accuracy = []
    sum = 0
    logreg = LogisticRegression()
    
    for i in range(k): # k-fold cross validation
        X_train, Y_train, indices = bootstrap_sample(X, Y, int(len(X) * 0.8)) # bootstrap sampling for 80% of the data
        X_test, Y_test = [], []
        score = 0 # accuracy score
        
        for j in range(len(indices)):
            if indices[j] == 0:
                X_test.append(X[j])
                Y_test.append(Y[j])
        
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test)
        length = len(Y_pred)
        
        for j in range(length):
            if Y_pred[j] == Y_test[j]: # if prediction is correct
                score += 1

        score /= length
        accuracy.append(score)
    
    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    logreg.fit(X, Y) # fit the model with all the data
    coeffs = logreg.coef_
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    
    return logreg, accuracy, coeffs

def decision_tree_model(X, Y, crit, k): # decision tree model
    accuracy = []
    sum = 0
    dectree = DecisionTreeClassifier(criterion = crit)
    
    for i in range(k): # k-fold cross validation
        X_train, Y_train, indices = bootstrap_sample(X, Y, int(len(X) * 0.8)) # bootstrap sampling for 80% of the data
        X_test, Y_test = [], []
        score = 0 # accuracy score
        
        for j in range(len(indices)):
            if indices[j] == 0:
                X_test.append(X[j])
                Y_test.append(Y[j])
        
        dectree.fit(X_train, Y_train)
        Y_pred = dectree.predict(X_test)
        length = len(Y_pred)
        
        for j in range(length):
            if Y_pred[j] == Y_test[j]: # if prediction is correct
                score += 1
        
        accuracy.append(score / length)
        
    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    dectree.fit(X, Y) # fit the model with all the data
    
    return dectree, accuracy
        
def random_forest_model(X, Y, k): # random forest model
    accuracy = []
    sum = 0
    randfor = RandomForestClassifier(bootstrap=False)
    
    for i in range(k): # k-fold cross validation
        X_train, Y_train, indices = bootstrap_sample(X, Y, int(len(X) * 0.8)) # bootstrap sampling for 80% of the data
        X_test, Y_test = [], []
        score = 0 # accuracy score
        
        for j in range(len(indices)):
            if indices[j] == 0:
                X_test.append(X[j])
                Y_test.append(Y[j])
                
        randfor = randfor.fit(X_train, Y_train)
        Y_pred = randfor.predict(X_test)
        length = len(Y_pred)
        
        for j in range(length):
            if Y_pred[j] == Y_test[j]: # if prediction is correct
                score += 1
        
        accuracy.append(score / length)

    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    randfor = randfor.fit(X, Y) # fit the model with all the data
    importance = randfor.feature_importances_
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    
    return randfor, accuracy, importance
    
def naive_bayes_model(X, Y, k): # naive bayes model
    accuracy = []
    sum = 0
    nb = GaussianNB()
    
    for i in range(k): # k-fold cross validation
        X_train, Y_train, indices = bootstrap_sample(X, Y, int(len(X) * 0.8)) # bootstrap sampling for 80% of the data
        X_test, Y_test = [], []
        score = 0 # accuracy score
        
        for j in range(len(indices)):
            if indices[j] == 0:
                X_test.append(X[j])
                Y_test.append(Y[j])
        
        nb.fit(X_train, Y_train)
        Y_pred = nb.predict(X_test)
        length = len(Y_pred)
        
        for j in range(length):
            if Y_pred[j] == Y_test[j]: # if prediction is correct
                score += 1
        
        accuracy.append(score / length)
        
    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    nb.fit(X, Y) # fit the model with all the data
    
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    
    print("Detailed accuracy: \n", detailed_accuracy(Y_test, Y_pred, fpr, tpr, auc))
    
    return nb, accuracy  

def dbscan_model(X, Y, k): # DBSCAN model  
    accuracy = []
    sum = 0
    dbscan = DBSCAN(eps=0.5, min_samples=6)
    
    for i in range(k): # k-fold cross validation
        X_train, Y_train, indices = bootstrap_sample(X, Y, int(len(X) * 0.8)) # bootstrap sampling for 80% of the data
        X_test, Y_test = [], []
        score = 0 # accuracy score
        
        for j in range(len(indices)):
            if indices[j] == 0:
                X_test.append(X[j])
                Y_test.append(Y[j])
    
        dbscan.fit(X_train, Y_train)
        Y_pred = dbscan.fit_predict(X_test)
        length = len(Y_pred)
        
        for j in range(length):
            if Y_pred[j] == Y_test[j]: # if prediction is correct
                score += 1
        
        accuracy.append(score / length)
        
    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    dbscan.fit(X, Y) # fit the model with all the data
    labels = dbscan.labels_ # cluster labels
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # number of clusters
    n_noise_ = list(labels).count(-1) # number of noise points
    homogenity = homogeneity_score(Y, labels) # homogenity score
    completeness = completeness_score(Y, labels) # completeness score
    v_measure = v_measure_score(Y, labels) # v-measure score
    adjusted_rand = adjusted_rand_score(Y, labels) # adjusted rand score
    adjusted_mutual_info = adjusted_mutual_info_score(Y, labels) # adjusted mutual info score
    silhouette = silhouette_score(X, labels) # silhouette score
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Homogenity: %0.3f' % homogenity)
    print('Completeness: %0.3f' % completeness)
    print('V-measure: %0.3f' % v_measure)
    print('Adjusted Rand Index: %0.3f' % adjusted_rand)
    print('Adjusted Mutual Information: %0.3f' % adjusted_mutual_info)
    print('Silhouette Coefficient: %0.3f' % silhouette)
    
    return dbscan, accuracy
        
def neural_network_model(X, Y, k): # neural network model
    accuracy = []
    sum = 0
    nn = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1, max_iter=1000, early_stopping=False)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # 80% training, 20% testing
    
    for i in range(k): # k-fold cross validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2) # 80% training, 20% validation   
        
        nn.fit(X_train, Y_train)
        Y_pred = nn.predict(X_test)
        
        score = nn.score(X_val, Y_val)
        accuracy.append(score)
        
    for i in range(len(accuracy)):
        sum += accuracy[i]
    accuracy = sum / len(accuracy) # average accuracy
    
    nn.fit(X_train, Y_train)
    plt.plot(nn.loss_curve_)
    plt.title('Neural Network Training Set Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training'], loc='upper left')
    plt.show()    
    
    Y_pred = nn.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    
    print("Detailed accuracy: \n", detailed_accuracy(Y_test, Y_pred, fpr, tpr, auc))
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
    
    nn.fit(X, Y) # fit the model with all the data
    return nn, accuracy

pd.options.mode.chained_assignment = None  # default='warn'
data = loadarff('chronic_kidney_disease.arff')
df = pd.DataFrame(data[0])
k = 10 # 10-fold cross validation
columns = df.columns
    

"data decoding"
df['sg'] = df['sg'].str.decode('utf-8')
df['al'] = df['al'].str.decode('utf-8')
df['su'] = df['su'].str.decode('utf-8')
df['rbc'] = df['rbc'].str.decode('utf-8')
df['pc'] = df['pc'].str.decode('utf-8')
df['pcc'] = df['pcc'].str.decode('utf-8')
df['ba'] = df['ba'].str.decode('utf-8')
df['htn'] = df['htn'].str.decode('utf-8')
df['dm'] = df['dm'].str.decode('utf-8')
df['cad'] = df['cad'].str.decode('utf-8')
df['appet'] = df['appet'].str.decode('utf-8')
df['pe'] = df['pe'].str.decode('utf-8')
df['ane'] = df['ane'].str.decode('utf-8')
df['class'] = df['class'].str.decode('utf-8')

"data preprocessing"
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0, '?':np.nan}) # yes/no/? -> 1/0/NaN
df[['rbc', 'pc']] = df[['rbc', 'pc']].replace(to_replace={'abnormal':1,'normal':0, '?':np.nan}) # abnormal/normal/? -> 1/0/NaN 
df[['pcc', 'ba']] = df[['pcc', 'ba']].replace(to_replace={'present':1,'notpresent':0, '?':np.nan}) # present/notpresent/? -> 1/0/NaN
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0, '?':np.nan}) # good/poor/? -> 1/0/NaN
df[['class']] = df[['class']].replace(to_replace={'ckd':1,'notckd':0, '?':np.nan}) # ckd/notckd/? -> 1/0/NaN
df[['sg', 'al', 'su']] = df[['sg', 'al', 'su']].replace(to_replace={'?':np.nan}) # ? -> NaN

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # convert to numeric

thrs = int(len(df.columns) / 3) # threshold for NaN values
for ind, row in df.iterrows(): # for each row
    sum = 0
    for col in df.columns:
        if math.isnan(df[col][ind]):
            sum += 1
    if sum > thrs:
        df.drop([ind], inplace=True)
df.reset_index(inplace=True, drop=True)

for col in df.columns:
    if col in ['sg', 'al', 'su', 'htn', 'dm', 'cad', 'pe', 'ane', 'rbc', 'pc', 'pcc', 'ba', 'appet']: # categorical data
        df[col] = df[col].fillna(df[col].mode()[0]) # fill NaN with mode
    else: # numerical data
        df[col] = df[col].fillna(df[col].mean()) # fill NaN with mean
        
for col in ['bgr', 'bu', 'sc', 'pot', 'wbcc']: # columns with high standard deviation
    average = df[col].mean()
    sd = df[col].std()
    for i in range(len(df[col])): # for each row
        if df[col][i] > average + 2 * sd:
            if col == 'wbcc':
                df[col][i] = df[col][i] / 5 # divide by 5
                continue
            df[col][i] = df[col][i] / 10 # divide by 10
        elif df[col][i] < average - 2 * sd:
            if col == 'wbcc':
                df[col][i] = df[col][i] * 5 # divide by 5
                continue
            df[col][i] = df[col][i] * 10 # multiply by 10


"data transformation"
df_x_lr, df_y_lr = normalize_transform(df.copy(), 'class') # normalized data for logistic regression
df_x_dt, df_y_dt = discretize_transform(df.copy(), 'class') # discretized data

"models"
id3_model, id3_accuracy = decision_tree_model(df_x_dt, df_y_dt, 'entropy', k)
cart_model, cart_accuracy = decision_tree_model(df_x_dt, df_y_dt, 'gini', k)

#print("ID3 Accuracy: ", id3_accuracy)
#print("CART Accuracy: ", cart_accuracy)

#print("------------------------------------")

logreg_model, logreg_accuracy, logreg_coeff = logistic_regression_model(df_x_lr, df_y_lr, k)
#print("Logistic Regression Accuracy: ", logreg_accuracy)
#print("Logistic Regression Coefficients: ", logreg_coeff)

#print("------------------------------------")

randfor_model, randfor_accuracy, randfor_importance = random_forest_model(df_x_dt, df_y_dt, k)
#print("Random Forest Accuracy: ", randfor_accuracy)

"""
feature_names = [f"{col}" for col in columns if col != 'class']
forest_importances = pd.Series(randfor_importance, index=feature_names)   
std = np.std([tree.feature_importances_ for tree in randfor_model.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()
"""

nb_model, nb_accuracy = naive_bayes_model(df_x_dt, df_y_dt, k)
print("Naive Bayes Accuracy: ", nb_accuracy)

pca = PCA(n_components=2) # PCA
df_x_pca = pca.fit_transform(df_x_lr) # dimensionality reduction to 2D for visualization

print(df_x_pca.shape)
print(df_x_lr.shape)

dbs_model, dbs_accuracy = dbscan_model(df_x_pca, df_y_lr, k)
print("DBSCAN Accuracy: ", dbs_accuracy)
print("------------------------------------")

# plot the clusters 
"""
plt.scatter(df_x_pca[:, 0], df_x_pca[:, 1], c=dbs_model.labels_, cmap='rainbow')
plt.title('DBSCAN Clusters')
plt.show()
"""

nn_model, nn_accuracy = neural_network_model(df_x_lr, df_y_lr, k)
print("Neural Network Accuracy: ", nn_accuracy)


plt.plot(nn_model.loss_curve_)
plt.title('Neural Network Test Set Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Test'], loc='upper left')
plt.show()
