from get_data import read_params, get_data,log
import argparse
import pandas as pd
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
class extract:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def extract_data(self,config_path):
        try:
            config = read_params(config_path)
            data_path=config["process_data"]["data"]
            df = pd.read_csv(data_path, sep=",")
            df.replace({"None":np.nan},inplace=True)
            log(self.file,"successfully replaced None values with missing values")
            x=df.drop("class",axis=1)
            y=df["class"]
            c=pd.DataFrame(x.isna().sum()/x.shape[0]*100,columns=['1'])
            d=c[c['1']>15]
            e=c[c['1']<15]
            d=d.T
            e=e.T
            imp = IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
            f=df[d.columns]
            g=imp.fit_transform(f)
            h=pd.DataFrame(g,columns=f.columns)
            h=h.astype(int)
            log(self.file,"handled columns missing value with more than 15%")
            i=df[e.columns]
            imp_mean = SimpleImputer( strategy='most_frequent')
            imp_mean.fit(i)
            j=pd.DataFrame(imp_mean.transform(i),columns=i.columns)
            j["cj_000"]=j["cj_000"].astype(float)
            j["dq_000"]=j["dq_000"].astype(float)
            j=j.astype(int)
            log(self.file,"handled missing values with less than 15%")
            df=pd.concat([h,j],axis=1)
            df["class"]=y
            df.to_csv("data/final/aps.csv",index=False)
            log(self.file,"finally data is ready for model training")
        except Exception as e:
            print(e)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    extract().extract_data(config_path=parsed_args.config)


