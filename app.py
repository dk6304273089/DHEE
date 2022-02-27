import streamlit as st
import joblib
import time
import numpy as np
st.title("SCANIA APS FAILURE")
try:
    col1,col2=st.columns(2)
    input1=col1.sidebar.number_input("Enter the value of aa_000",min_value=0,max_value=2746564)
    input2=col2.sidebar.number_input("Enter the value of ag_002",min_value=0,max_value=10552856)
    col3,col4=st.columms(2)
    input3=col3.sidebar.number_input("Enter the value of cs_004",min_value=0,max_value=74860628)
    input4=col4.sidebar.number_input("Enter the value of ay_008",min_value=0,max_value=104566992)
    col5,col6=st.columms(2)
    input5=col5.sidebar.number_input("Enter the value of dn_000",min_value=0,max_value=2924584)
    input6=col6.sidebar.number_input("Enter the value of cj_000",min_value=0,max_value=60949671)
    ok = st.button("SUBMIT")
#aa_000 = int(st.sidebar.number_input("Enter the value of aa_000",min_value=0,max_value=2746564))
#ag_002=int(st.sidebar.number_input("Enter the value of ag_002",min_value=0,max_value=10552856))
#cs_004=int(st.sidebar.number_input("Enter the value of cs_004",min_value=0,max_value=74860628))
#ay_008=int(st.sidebar.number_input("Enter the value of ay_008",min_value=0,max_value=104566992))
#dn_000=int(st.sidebar.number_input("Enter the value of dn_000",min_value=0,max_value=2924584))
#cj_000=int(st.sidebar.number_input("Enter the value of cj_000",min_value=0,max_value=60949671))
    if ok:
        values=[[input1,input2,input3,input4,input5,input6]]
        model=joblib.load("saved_models/model.joblib")
        model1=joblib.load("saved_models/sc.joblib")
        transform=model1.transform(values)
        c=np.where(model.predict_proba(transform)[:,1]>0.9,1,0)
    if c==[0]:
        st.success("The Failure is not related to air pressure system")
    else:
        st.subheader("The Failure is related to air pressure system")
except ValueError:
    st.error("Please enter a valid input")
   

