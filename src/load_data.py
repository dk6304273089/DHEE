from get_data import read_params, get_data,log
import argparse
class data:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def load_data(self,config_path):
        try:
            config=read_params(config_path)
            df=get_data(config_path)
            df.drop(["af_000","ag_004","ag_007","ah_000","an_000","ao_000","ap_000","ba_002","ba_003","ba_004","bb_000","bg_000","bh_000","bt_000","bu_000","bv_000","bx_000","by_000","cc_000","ci_000","cq_000","cs_005"],axis=1,inplace=True)
            df.drop(["az_005","ba_000","ba_001","ba_005","ba_006","cn_001","cn_002","cn_003","cn_004","cn_005","ee_001","ee_002","ee_003","ag_005"],axis=1,inplace=True)
            df.drop(["dc_000","cv_000","bl_000","ee_000"],axis=1,inplace=True)
            df.drop(["am_0","aq_000","ay_003","cz_000"],axis=1,inplace=True)
            df.drop(["cm_000","bj_000","ck_000","do_000","dm_000","dt_000","ec_00","ad_000","cl_000","cf_000"],axis=1,inplace=True)
            df.drop(['ab_000','bm_000','bn_000','bo_000','bp_000','bq_000','cr_000','br_000','ind'],axis=1,inplace=True)
            log(self.file,"successfully removed unwanted columns")
            df.replace(to_replace =["neg","pos"],value =[0,1],inplace=True)
            df["class"]=df["class"].astype(int)
            path=config["process_data"]["data"]
            df.to_csv("data/processed/aps.csv",index=False)
            log(self.file,"successfully saved the processed file")
        except Exception as e:
            print(e)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    data().load_data(config_path=parsed_args.config)