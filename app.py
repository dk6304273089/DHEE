import streamlit as st
from prediction.data import read_data,drop_data
import joblib
import time
st.title("SCANIA APS FAILURE")
name = st.text_input("ENTER THE INPUT PATH FILE")
name1=st.text_input("ENTER THE OUTPUT FOLDER PATH FOLDER")
print(name)
print(name1)
ok = st.button("SUBMIT")
if ok:
    df=read_data(r"{}".format(name))
    start = time.time()
    z=df
    df=drop_data(df)
    model=joblib.load("saved_models/model.joblib")
    z["class"]=model.predict(df)
    end = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    z.to_csv( name1 + '\\output_'  + '.csv' , index=False )
    st.subheader("PROCESS COMPLETED. PLEASE CHECK OUTPUT DIRECTORY. TOTAL TIME TAKEN: {}".format(end-start))

