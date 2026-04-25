# ZAINAB
# CF-25097

#importing libraries
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#extracting all joblib files
rscore = joblib.load("R2score.pkl")
error=joblib.load("meanabserror.pkl")
cols=joblib.load("columns.pkl")
model=joblib.load("regressionmodel.pkl")
ypred=joblib.load("ypredictedvals.pkl")
y_test=joblib.load("ytestvals.pkl")

#setting page configuration, title etc
st.set_page_config(page_title="Price Predictor",page_icon="📍",layout="wide")
st.title("***:gray[REAL ESTATE PRICE PREDICTOR]***")
st.markdown('Author: Zainab CF-25097')

#printing r2 score and mean absolute error on app in 2 columns
r2,mae,empty = st.columns(3)
with r2:
    st.markdown("#### **Model Reliability** (R2 SCORE):")
    st.write(f"{rscore:,.2f}")

with mae:
    st.markdown("#### **Mean Absolute Error of the Model:**")
    st.write(f"{error:,.2f}")

#changing names of data frame columns to show in app
location=[i.replace('location_','')for i in cols if 'location_' in i]
furnishingstat=[i.replace('Furnishing_','')for i in cols if 'Furnishing_' in i]

#creating data frame to store input values and then predict
values=pd.DataFrame(0,index=[0], columns = cols)

#creating 2 tabs on app for prediction and analytics
pred, analytics = st.tabs(["**Prediction**","**Analytics**"])
with pred:
    st.markdown("## :gray[Enter the details then click \"Predict\".]")

    #creating columns to have input
    loct,furnt,bth,flr=st.columns(4,border=True)
    with loct:
        selectedloc=st.selectbox("Select Location:",location)
    with furnt:
        selectedfurnishingstat=st.selectbox("Select Furnishing Status:",furnishingstat)
    with bth:
        bathroom = st.slider("Select Number of Bathrooms:",min_value=0,max_value=5)
    with flr:
        floorx=st.number_input("Floor Number",min_value=0,max_value=50,value=1, step=1)
        floory= st.number_input("Total Floors in Building",max_value=50,value=floorx,step=1)
        if floorx>floory:
            raise st.error("floor exceeding range")

    #creating prediction button
    button=st.button("PREDICT NOW")

#creating side bar to print the chosen inputs
st.sidebar.header("***PROPERTY FEATURES***")

with st.sidebar:
    st.write("Features you entered:")
    
    st.markdown("## **LOCATION:**")
    st.write(selectedloc.title())

    st.markdown("## **FURNISHING STATUS:**")
    st.write(selectedfurnishingstat)

    st.markdown("## **BATHROOMS:**")
    st.write(f"{bathroom}")

    st.markdown("## **FLOOR:**")
    st.write(f"Current floor: {int(floorx)}")
    st.write(f"Total floors in building: {int(floory)}")

#changing column names for data frame according to selected values
locationcolumn=f'location_{selectedloc}'
furcolumn=f'Furnishing_{selectedfurnishingstat}'
floorcolumn=f'Floor_{floorx} out of {floory}'

#updating values in data frame
values['Bathroom']=bathroom

if locationcolumn in values.columns:
    values[locationcolumn]=1

if furcolumn in values.columns:
    values[furcolumn]=1

if floorcolumn in values.columns:
    values[floorcolumn]=1
else:
    floorratio = floorx/floory
    values['floorratio']=floorratio
    
#predicting price
log_price=model.predict(values)

#removing log
finalp=np.exp(log_price)[0]

#converting into crore or lac for convenience
if finalp>= 10000000:
    finalp= f"{finalp/10000000:.2f} Crore"
else:
    finalp=f"{finalp/100000:.2f} Lac"

#printing the predicted price when button is clicked
if button:
    st.markdown("#### **ESTIMATED MARKET PRICE:**")
    st.success(f"₹ {finalp}")
    upperbound=np.exp(log_price+error)[0]
    lowerbound=np.exp(log_price-error)[0]

    #printing analysis
    st.markdown("#### *PRICE RANGE ANALYSIS*")
    st.write(f"We are sure price is ranging between ₹ {round(lowerbound)} and ₹ {round(upperbound)}")

#analytics tab
with analytics:
    st.markdown("#### Regression Plot:")   
    st.write("This plot visualizes the deviation between actual values of data and my model's guesses.")

    fig,ax=plt.subplots()
    ax.scatter(y_test,ypred)
    ax.set_xlabel("Original Price")
    ax.set_ylabel("Estimated Price")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    st.pyplot(fig)
