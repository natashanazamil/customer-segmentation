#%% imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
import datetime
import os
#from tensorflow.keras.utils import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

#%% 1. Data Loading
df = pd.read_csv('train.csv')


#%% 2. EDA
df.info()
df.describe().T

#%% 3. Data Cleaning
df.isna().sum()

cat_col = df.columns[df.dtypes == 'object']
con_col = df.columns[(df.dtypes == 'float64')|(df.dtypes == 'int64')]

#%%
df['Profession'].unique()
no =  (df['Profession'] == 'Homemaker') | (df['Profession'] == 'Lawyer') | (df['Profession'] == 'Marketing')|(df['Profession'] == 'Executive')
df['Profession'] = np.where(no, np.nan, df['Profession'])

no2 = (df['Var_1'] == 'Cat_1') | (df['Var_1'] == 'Cat_5')
df['Var_1'] = np.where(no2, np.nan, df['Var_1'])
#%%
df['Age'].boxplot()

#%%
mode1 = df['Ever_Married'].mode().iloc[0]
df['Ever_Married'].fillna(mode1, inplace=True)

mode2 = df['Graduated'].mode().iloc[0]
df['Graduated'].fillna(mode2, inplace=True)

mode3 = df['Profession'].mode().iloc[0]
df['Profession'].fillna(mode3, inplace=True)

mode4 = df['Var_1'].mode().iloc[0]
df['Var_1'].fillna(mode4, inplace=True)

for con in con_col:
    df[con] = df[con].fillna(df[con].median())

#%%
df.duplicated().sum()
df.nunique()

#%%

for cat in cat_col:
    plt.figure()
    sns.displot(df[cat])
    plt.show()

for con in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis=1),df['Segmentation'])
    print(con)
    print(lr.score(np.expand_dims(df[con], axis=1),df['Segmentation']))

#%%
le = LabelEncoder()
df['Gender']= le.fit_transform(df['Gender'])
df['Ever_Married'] = le.fit_transform(df['Ever_Married'])
df['Graduated'] = le.fit_transform(df['Graduated'])
df['Profession'] = le.fit_transform(df['Profession'])
df['Spending_Score'] = le.fit_transform(df['Spending_Score'])
df['Var_1'] = le.fit_transform(df['Var_1'])
df['Segmentation'] = le.fit_transform(df['Segmentation'])


#%% 4. Feature Selection
y = df['Segmentation']
X = df.drop(labels=['Segmentation', 'ID'], axis=1)

lr = LogisticRegression()

for cat in cat_col:
    lr.fit(np.expand_dims(y, axis=-1), 
            np.expand_dims(df[cat], axis = -1))
    print(cat)
    print(lr.score(np.expand_dims(y, axis=-1), 
            np.expand_dims(df[cat], axis = -1)))

#%%
df_con = df[con_col]

plt.figure(figsize=(20,20))
sns.heatmap(df_con.corr(), annot=True, cmap=plt.cm.Reds)
plt.show()

#%% 5. Data Preprocessing


ss = StandardScaler()
X = ss.fit_transform(X)

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y, axis=-1))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


#%% 6. Model Development
input_shape = np.shape(X_train)[1:]
output_shape = np.shape(y_train)[1:][0]


model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(output_shape, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics='accuracy')


log_dir = os.path.join(os.getcwd(),datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)

# early stopping callback
es_callback = EarlyStopping(monitor='loss',patience=3)
hist = model.fit(X_train,y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=100, 
                    callbacks=[tb_callback, es_callback])


#%% 7. Model Analysis
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

#%%
cr = classification_report(y_test, y_pred)
print(cr)

# %%
