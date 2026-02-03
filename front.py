import numpy as np
import pandas as pd
import streamlit as st
import pickle 

filename='/Users/adityayadav/Desktop/ DS  Nareshit/All_Projects/Prediction_Height/LR_model.pkl'
with open(filename, 'rb') as f:
    loaded_model=pickle.load(f)

st.title('Prediction of the weight through the Height')

st.write('By the calculation of height and predicte the weight')

preprocessor=st.number_input('Enter the number:', min_value=0.0)

if st.button('Predict'):
    height=np.array(preprocessor).reshape(-1,1)
    predictd_weight= loaded_model.predict(height)
    st.markdown(predictd_weight[0,0])

st.write('Successfully! Created the webpage')