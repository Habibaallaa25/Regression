import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def preprocess(name):
    data = pd.read_csv(name)


    df=data.drop(columns=['X1','X7'])


    df['X2'].fillna(method ='bfill', inplace = True)
    df['X4']=df['X4'].replace(0,nan)
    df['X4'].fillna(method='bfill',inplace=True)
    df['X9'].fillna(method='bfill',inplace=True)


    values=df['X3'].values
    for i in range(values.size):
        if (values[i][0]=='L'or values[i][0]=='l'):
            values[i]='Low Fat'
        elif(values[i][0]=='R'or values[i][0]=='r'):
            values[i]='Regular'

    df['X3']=values

    label_encoder=preprocessing.LabelEncoder()
    df['X3']=label_encoder.fit_transform(df['X3'])
    df['X5']=label_encoder.fit_transform(df['X5'])
    df['X9']=label_encoder.fit_transform(df['X9'])
    df['X10']=label_encoder.fit_transform(df['X10'])
    df['X11']=label_encoder.fit_transform(df['X11'])


    df2 = df.drop(columns=['X3','X5','X8', 'X2'])
    X=df2[['X4','X9','X10','X11']]
    norm=MinMaxScaler().fit(X)
    df2[['X4','X9','X10','X11']]=norm.transform(df[['X4','X9','X10','X11']])
    return df2

df=preprocess('train.csv')
x = df[['X4','X6','X9','X10','X11']].values
y = df['Y'].values
df_test=preprocess('test.csv')
x_test = df_test.values
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)


model=DecisionTreeRegressor(max_depth=30, min_samples_leaf=120)
model.fit(X_train,Y_train)
pred=model.predict(X_test)
print(mean_absolute_error(pred,Y_test))
y_pred=model.predict(x_test)
# pd.DataFrame(y_pred,columns=['predictions']).to_csv('predictionDecisionTree.csv')

plot=pd.DataFrame()
plot['Target']=Y_test
plot['Predictions']=pred

sns.lmplot('Target','Predictions',data=plot,height=6,aspect=2,line_kws={'color':'green'},scatter_kws={'alpha':0.4,'color':'blue'})
plt.title('Decision Tree Regression \n Mean Error: {0:.2f}'.format(mean_absolute_error(Y_test, pred)),size=25)
plt.show()