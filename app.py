import streamlit as st
from prediction.data import read_data,drop_data
import joblib
st.title("SCANIA APS FAILURE")
name = st.text_input("ENTER THE INPUT PATH FILE")
name1=st.text_input("ENTER THE OUTPUT FOLDER PATH FOLDER")
print(name)
print(name1)
ok = st.button("SUBMIT")
if ok:
    df=read_data(r"{}".format(name))
    z=df
    df=drop_data(df)
    model=joblib.load(r"C:\Users\Dheeraj kumar\OneDrive\Desktop\aps\saved_models\model.joblib")
    z["class"]=model.predict(df)
    z.to_csv( name1 + '\\output_'  + '.csv' , index=False )
    st.subheader("the Process Complete. Please check Output Directory")

