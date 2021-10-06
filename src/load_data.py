from get_data import read_params, get_data,log
import argparse
class data:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def load_data(self,config_path):
        try:
            config=read_params(config_path)
            df=get_data(config_path)
            df=df[["aa_000","ag_002","cs_004","ay_008","dn_000","cj_000","class"]]
            log(self.file,"successfully added columns which are required")
            df.replace(to_replace =["neg","pos"],value =[0,1],inplace=True)
            df["class"]=df["class"].astype(int)
            df.to_csv("data/processed/aps.csv",index=False)
            log(self.file,"successfully saved the processed file")
        except Exception as e:
            print(e)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    data().load_data(config_path=parsed_args.config)