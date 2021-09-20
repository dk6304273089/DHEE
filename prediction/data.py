import pandas as pd
import sklearn
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

def read_data(path):
    df=pd.read_csv(r"{}".format(path),skiprows=20)
    return df

def log(file_object, log_message):
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        file_object.write(
            str(date) + "/" + str(current_time) + "\t\t" + log_message +"\n")

file=open("Training_logs/Training_log.txt","a+")

def drop_data(df):
    df.drop(["af_000","ag_004","ag_007","ah_000","an_000","ao_000","ap_000","ba_002","ba_003","ba_004","bb_000","bg_000","bh_000","bt_000","bu_000","bv_000","bx_000","by_000","cc_000","ci_000","cq_000","cs_005"],axis=1,inplace=True)
    log(file,"Dropped 1st columns")
    df.drop(["az_005","ba_000","ba_001","ba_005","ba_006","cn_001","cn_002","cn_003","cn_004","cn_005","ee_001","ee_002","ee_003","ag_005"],axis=1,inplace=True)
    log(file,"Dropped 2nd columns")
    df.drop(["dc_000","cv_000","bl_000","ee_000"],axis=1,inplace=True)
    log(file,"Dropped 3rd columns")
    df.drop(["am_0","aq_000","ay_003","cz_000"],axis=1,inplace=True)
    log(file,"Dropped 4th columns")
    df.drop(["cm_000","bj_000","ck_000","do_000","dm_000","dt_000","ec_00","ad_000","cl_000","cf_000"],axis=1,inplace=True)
    log(file,"Dropped 5th columns")
    df.drop(['ab_000','bm_000','bn_000','bo_000','bp_000','bq_000','cr_000','br_000','class'],axis=1,inplace=True)
    log(file,"Dropped all columns")
    df.replace({"na":np.nan},inplace=True)
    log(file,"Done replaced nan")
    c=pd.DataFrame(df.isna().sum()/df.shape[0]*100,columns=['1'])
    print(c.head())
    d=c[c['1']>15]
    e=c[c['1']<15]
    d=d.T
    e=e.T
    log(file,"transformation done")
    lr=LinearRegression()
    imp = IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    f=df[d.columns]
    g=imp.fit_transform(f)
    h=pd.DataFrame(g,columns=f.columns)
    h=h.astype(int)
    i=df[e.columns]
    log(file,"Done iterative imputer")
    imp_mean = SimpleImputer( strategy='most_frequent')
    imp_mean.fit(i)
    j=pd.DataFrame(imp_mean.transform(i),columns=i.columns)
    j=j.astype(float)
    j=j.astype(int)
    df=pd.concat([h,j],axis=1)
    log(file,"successfully completed")
    return df


