import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from get_data import read_params, get_data,log
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import os
import json
class train:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def data_read(self,config_path="config.yaml"):
        try:
            config = read_params(config_path)
            data_path=config["final_data"]["data"]
            model_dir = config["model_dir"]
            df = pd.read_csv(data_path, sep=",")
            return df,model_dir
        except Exception as e:
            print(e)
        

    def train(self,config_path):
        try:
            config = read_params(config_path)
            df,model_dir=self.data_read()
            x=df.drop("class",axis=1)
            y=df["class"]
            sc=StandardScaler()
            X=sc.fit_transform(x)
            base_score=config["estimators"]["xgboost"]["params"]["base_score"]
            booster=config["estimators"]["xgboost"]["params"]["booster"]
            colsample_bylevel=config["estimators"]["xgboost"]["params"]["colsample_bylevel"]
            colsample_bynode=config["estimators"]["xgboost"]["params"]["colsample_bynode"]
            colsample_bytree=config["estimators"]["xgboost"]["params"]["colsample_bytree"]
            gamma=config["estimators"]["xgboost"]["params"]["gamma"]
            gpu_id=config["estimators"]["xgboost"]["params"]["gpu_id"]
            importance_type=config["estimators"]["xgboost"]["params"]["importance_type"]
            learning_rate=config["estimators"]["xgboost"]["params"]["learning_rate"]
            max_delta_step=config["estimators"]["xgboost"]["params"]["max_delta_step"]
            max_depth=config["estimators"]["xgboost"]["params"]["max_depth"]
            min_child_weight=config["estimators"]["xgboost"]["params"]["min_child_weight"]
            n_estimators=config["estimators"]["xgboost"]["params"]["n_estimators"]
            n_jobs=config["estimators"]["xgboost"]["params"]["n_jobs"]
            num_parallel_tree=config["estimators"]["xgboost"]["params"]["num_parallel_tree"]
            random_state=config["estimators"]["xgboost"]["params"]["random_state"]
            reg_alpha=config["estimators"]["xgboost"]["params"]["reg_alpha"]
            reg_lambda=config["estimators"]["xgboost"]["params"]["reg_lambda"]
            scale_pos_weight=config["estimators"]["xgboost"]["params"]["scale_pos_weight"]
            subsample=config["estimators"]["xgboost"]["params"]["subsample"]
            tree_method=config["estimators"]["xgboost"]["params"]["tree_method"]
            validate_parameters=config["estimators"]["xgboost"]["params"]["validate_parameters"]
            xg=XGBClassifier(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
              colsample_bynode=colsample_bynode, colsample_bytree=colsample_bytree, gamma=gamma, gpu_id=gpu_id,
              importance_type=importance_type,
              learning_rate=learning_rate, max_delta_step=max_delta_step, max_depth=max_depth,
              min_child_weight=min_child_weight,
              n_estimators=n_estimators, n_jobs=n_jobs, num_parallel_tree=num_parallel_tree, random_state=random_state,
              reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, subsample=subsample,
              tree_method=tree_method, validate_parameters=validate_parameters)
            X_train,X_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
            log(self.file,"successfully applied smote")
            log(self.file,"training started")
            xg.fit(X_train_smote,y_train_smote)
            predicted_qualities = np.where(xg.predict_proba(X_test)[:,1]>0.6,1,0)    


            scores_file = config["reports"]["scores"]
            params_file = config["reports"]["params"]

            with open(scores_file, "w") as f:
                scores = {
                    "accuracy_score": accuracy_score(y_test,predicted_qualities),
                    "roc_auc_score": roc_auc_score(y_test,predicted_qualities),
                    "f1_score": f1_score(y_test,predicted_qualities)
                    }
                json.dump(scores, f, indent=4)

            with open(params_file, "w") as f:
                params = {
                    "base_score": base_score,
                    "booster": booster,
                    "colsample_bylevel": colsample_bylevel,
                    "colsample_bynode": colsample_bynode,
                    "colsample_bytree": colsample_bytree,
                    "gamma": gamma,
                    "gpu_id": gpu_id,
                    "importance_type": importance_type,
                    "learning_rate": learning_rate,
                    "max_delta_step": max_delta_step,
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "n_estimators": n_estimators,
                    "n_jobs": n_jobs,
                    "num_parallel_tree": num_parallel_tree,
                    "random_state": random_state,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                    "scale_pos_weight": scale_pos_weight,
                    "subsample": subsample,
                    "tree_method": tree_method,
                    "validate_parameters": validate_parameters,
                }
                json.dump(params, f, indent=4)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.joblib")
            log(self.file,"model dumped successfully")
            joblib.dump(xg, model_path)
        except Exception as e:
            print(e)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    train().train(config_path=parsed_args.config)