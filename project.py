#ZAINAB
#CF-25097

#importing libraries needed
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np 
import joblib

#reading the data
data = pd.read_csv("house_prices.csv")

#print(data.info())

#deleting irrelevant columns from data set
data.drop(columns=['Title','Description','Carpet Area','Status','Transaction','facing','overlooking','Society'
,'Car Parking','Ownership','Super Area','Dimensions','Plot Area','Balcony'],inplace=True)

#print(data.isnull().sum())

#removing and updating empty rows from data set
data.dropna(subset=['Bathroom'],inplace=True)
data.dropna(subset=['Price (in rupees)'],inplace=True)
data['Furnishing']=data['Furnishing'].fillna('Unknown')
data['Floor']=data['Floor'].fillna('Unknown')

#converting amount from lac and cr 
def convert(i):
    if 'Lac' in i:
        i=i.replace('Lac','').strip()
        return float(i)*100000
    elif 'Cr' in i:
        i=i.replace('Cr','').strip()
        return float(i) * 10000000
    else:
        raise ValueError

data['Amount(in rupees)']=data['Amount(in rupees)'].apply(convert)

#converting data types
def convertint(i):
    if isinstance(i,str) and '>' in i:
        return int(i.replace('>',''))
    else: 
        return int(i)

data['Bathroom']=data['Bathroom'].apply(convertint)
# data.Bathroom = data.Bathroom.astype(int)  extra line

# data= pd.get_dummies(data, columns=['location'])

#adding a column "floor ratio"
data['current'], data['total']= data['Floor'].str.split(' out of ', expand=True)
data['current'] = pd.to_numeric(data['current'], errors='coerce')
data['total'] = pd.to_numeric(data['total'], errors='coerce')
data['floorratio']=data['current']/data['total']
data=data.drop(columns=['current','total'])

data = pd.get_dummies(data)

# print(data.isnull().sum())
# print(data.head(15))
# print(data.info())

#linear regression model
x = data.drop('Amount(in rupees)',axis = 1)
y = np.log(data['Amount(in rupees)'])

#splitting data set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)
model = LinearRegression()
model.fit(x_train,y_train)

ypredicted =  model.predict(x_test)
r2sc = r2_score(y_test,ypredicted)
mae = mean_absolute_error(y_test,ypredicted)

# plt.scatter(y_test,ypredicted)
# plt.xlabel("original price")
# plt.ylabel("estimated price")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
# plt.show()

# extracting the model to use in project
joblib.dump(model,"regressionmodel.pkl")

# extracting x columns, y test & predicted values, r2 score and mae
cols=list(x.columns)
joblib.dump(cols,"columns.pkl")

joblib.dump(r2sc,"R2score.pkl")
joblib.dump(mae,"meanabserror.pkl")

joblib.dump(y_test,"ytestvals.pkl")
joblib.dump(ypredicted,"ypredictedvals.pkl")