#%% Imports
import pandas as pd
import os
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.impute import KNNImputer

CSV_PATH = os.path.join(os.getcwd(), 'dataset','train.csv')


#%% func
def cramers_corrected_stat(confusion_matrix):

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher,

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% 1: Data Loading
df = pd.read_csv(CSV_PATH)

#%% 2: EDA
df.info()
df.describe().T

# categorical
cat_col = list(df.columns[df.dtypes=='object'])
cat_col.append('Family_Size')

con_col = list(df.columns[(df.dtypes=='int64')|(df.dtypes=='float64')])
con_col.remove('Family_Size')
con_col.remove('ID')

df.groupby(['Segmentation','Profession']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Gender']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Ever_Married']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Spending_Score']).agg({'Segmentation':'count'}).plot(kind='bar')

#%%
for con in con_col:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

for cat in cat_col:
    plt.figure()
    sns.countplot(x=df[cat])
    plt.show()

#%% 3: Data Cleaning
df.isna().sum()

df = df.drop(labels=['ID'], axis=1)

#%% label encoding for features & target
le = LabelEncoder()

for cat in cat_col:
    if cat == 'Family_Size':
        continue
    else:
        temp = df[cat]
        temp[df[cat].notnull()] = le.fit_transform(temp[df[cat].notnull()])
        df[cat] = pd.to_numeric(df[cat], errors='coerce')
        save_path = os.path.join(os.getcwd(),'model',cat+'_encoder.pkl')
        with open(save_path,'wb') as f:
            pickle.dump(le,f)

#%% isna using KNN imputation
column_names = df.columns

ki = KNNImputer(n_neighbors=5)
df = ki.fit_transform(df) # convert df to nparray

#%% to convert back to sf format
df = pd.DataFrame(df,columns=column_names)
df.isna().sum()


#%% 4: Feature Selection
# continuous vs categorical [target]
for con in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis=-1),df['Segmentation'])
    print(con)
    print(lr.score(np.expand_dims(df[con], axis=-1),df['Segmentation']))

# categorical vs categorical [target]
for cat in cat_col:
    cm = pd.crosstab(df[cat],df['Segmentation']).to_numpy()
    print(cat)
    print(cramers_corrected_stat(cm))


#%% 5: Data Preprocessing
X = df.drop(labels=['Segmentation'], axis=1)
y = df['Segmentation']

X_test, X_train, y_test, y_train = train_test_split(X,y, train_size=0.7, shuffle=True, random_state=123)


# %% Model Development
# KNN, Decision Tree, Random Forest, SVC, Logistic Regression

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


#KNN
steps_mms_knn = Pipeline([('mms', MinMaxScaler()),
                        ('knn',KNeighborsClassifier())])

steps_ss_knn = Pipeline([('ss', StandardScaler()),
                        ('knn',KNeighborsClassifier())])


#DT
steps_mms_dt = Pipeline([('mms', MinMaxScaler()),
                        ('dt',DecisionTreeClassifier())])

steps_ss_dt = Pipeline([('ss', StandardScaler()),
                        ('dt',DecisionTreeClassifier())])


#RFC
steps_mms_rf = Pipeline([('mms', MinMaxScaler()),
                         ('rf',RandomForestClassifier())])

steps_ss_rf = Pipeline([('ss', StandardScaler()),
                        ('rf',RandomForestClassifier())])


#SVC
steps_mms_svc = Pipeline([('mms', MinMaxScaler()),
                        ('svc',SVC())])

steps_ss_svc = Pipeline([('ss', StandardScaler()),
                        ('svc',SVC())])


#LR
steps_mms_lr = Pipeline([('mms', MinMaxScaler()),
                        ('lr',LogisticRegression())])

steps_ss_lr = Pipeline([('ss', StandardScaler()),
                        ('lr',LogisticRegression())])


pipelines = [steps_mms_knn,steps_ss_knn,
            steps_mms_dt, steps_ss_dt,
            steps_mms_rf, steps_ss_rf,
            steps_mms_svc, steps_ss_svc,
            steps_mms_lr, steps_ss_lr]

for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy = 0
for i, model in enumerate(pipelines):
    if model.score(X_test, y_test) > best_accuracy:
        best_accuracy = model.score(X_test, y_test)
        best_pipeline = model

# %%
print('The best piepine is {} with accuracy of {}'.format(best_pipeline, best_accuracy))

# %%
