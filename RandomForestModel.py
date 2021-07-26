import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def preprocess(name):
    data = pd.read_csv(name)


    df=data.drop(columns=['X1','X7'])
    values=df['X3'].values
    for i in range(values.size):
        if (values[i][0]=='L'or values[i][0]=='l'):
            values[i]='Low Fat'
        elif(values[i][0]=='R'or values[i][0]=='r'):
            values[i]='Regular'

    df['X3']=values

    df['X2'].fillna(method ='bfill', inplace = True)


    df['X4']=df['X4'].replace(0,nan)
    df['X4'].fillna(method='bfill',inplace=True)
    df['X9'].fillna(method='bfill',inplace=True)

    label_encoder=preprocessing.LabelEncoder()
    df['X3']=label_encoder.fit_transform(df['X3'])
    df['X5']=label_encoder.fit_transform(df['X5'])
    df['X9']=label_encoder.fit_transform(df['X9'])
    df['X10']=label_encoder.fit_transform(df['X10'])
    df['X11']=label_encoder.fit_transform(df['X11'])


    print(df.corr())
    df2 = df.drop(columns=['X3', 'X2','X5','X8'])
    print(df.describe())
    X=df2[['X4','X9','X10','X11']]
    norm=MinMaxScaler().fit(X)
    df2[['X4','X9','X10','X11']]=norm.transform(df[['X4','X9','X10','X11']])
    # print(df2.corr())

    return df2

df=preprocess('train.csv')
x = df[['X4','X6','X9','X10','X11']].values
y = df['Y'].values
df_test=preprocess('test.csv')
x_test = df_test.values
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)

model = RandomForestRegressor(n_estimators=39, max_depth=5)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

y_out2=model.predict(x_test)
err=mean_absolute_error(y_pred,Y_test)
print(err)
# pd.DataFrame(y_out2,columns=['Y']).to_csv('predictionRandomForest.csv')

plot=pd.DataFrame()
plot['Target']=Y_test
plot['Predictions']=y_pred

sns.lmplot('Target','Predictions',data=plot,height=6,aspect=2,line_kws={'color':'green'},scatter_kws={'alpha':0.4,'color':'blue'})
plt.title('Random Forest Regression \n Mean Error: {0:.2f}'.format(mean_absolute_error(Y_test, y_pred)),size=25)
plt.show()