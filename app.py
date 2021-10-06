import streamlit as st
import joblib
import time
import numpy as np
st.title("SCANIA APS FAILURE")
aa_000 = int(st.sidebar.number_input("Enter the value of aa_000",min_value=0,max_value=2746564))
ag_002=int(st.sidebar.number_input("Enter the value of ag_002",min_value=0,max_value=10552856))
cs_004=int(st.sidebar.number_input("Enter the value of cs_004",min_value=0,max_value=74860628))
ay_008=int(st.sidebar.number_input("Enter the value of ay_008",min_value=0,max_value=104566992))
dn_000=int(st.sidebar.number_input("Enter the value of dn_000",min_value=0,max_value=2924584))
cj_000=int(st.sidebar.number_input("Enter the value of cj_000",min_value=0,max_value=60949671))
ok = st.button("SUBMIT")
if ok:
    values=[[aa_000,ag_002,cs_004,ay_008,dn_000,cj_000]]
    model=joblib.load("saved_models/model.joblib")
    model1=joblib.load("saved_models/sc.joblib")
    transform=model1.transform(values)
    c=np.where(model.predict_proba(transform)[:,1]>0.9,1,0)
    if c==[0]:
        st.subheader("The Failure is not related to air pressure system")
    else:
        st.subheader("The Failure is related to air pressure system")

   

