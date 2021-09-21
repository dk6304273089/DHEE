#READING THE DATA FROM DATABASE
import os
from datetime import datetime
import yaml
import pandas as pd

import argparse
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
def read_params(config_path):
        with open(config_path) as yaml_file:
            config=yaml.safe_load(yaml_file)
        return config
def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = config["load_data"]["data"]
    df = pd.read_csv(data_path, sep=",")
    return df
def log(file_object, log_message):
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        file_object.write(
            str(date) + "/" + str(current_time) + "\t\t" + log_message +"\n")

class Data_extraction:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    
    def get(self,config_path):
        try:
            config=read_params(config_path)
            c=config["data_source"]["h1_data"]
            cloud_config= {
            'secure_connect_bundle': "{}".format(c)}
            auth_provider = PlainTextAuthProvider('dk6304273089@gmail.com', 'Dheerajkumar@123')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,idle_heartbeat_interval=10)
            session = cluster.connect()
            query = "SELECT * FROM aps1.aps1";
            log(self.file,"Data 1 extraction started")
            df1 = pd.DataFrame(list(session.execute(query)))
            df1.rename({"field_57_":"class"},axis=1,inplace=True)
            df1=df1.sort_values(by=['ind'])
            log(self.file,"Data 1 extraction completed")

            data2=config["data_source"]["h2_data"]
            cloud_config= {
            'secure_connect_bundle': "{}".format(data2)}
            auth_provider = PlainTextAuthProvider('dk6304273089@gmail.com', 'Dheerajkumar@123')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,idle_heartbeat_interval=15)
            session = cluster.connect()
            query = "SELECT * FROM h22.aps2";
            log(self.file,"Data 2 extraction started")
            df2 = pd.DataFrame(list(session.execute(query)))
            df2=df2.sort_values(by=['ind'])
            df2.drop("ind",axis=1,inplace=True)
            log(self.file,"Data 2 extraction completed")

            data3=config["data_source"]["h3_data"]
            cloud_config= {
            'secure_connect_bundle': "{}".format(data3)}
            auth_provider = PlainTextAuthProvider('dk6304273089@gmail.com', 'Dheerajkumar@123')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,idle_heartbeat_interval=25)
            session = cluster.connect()
            query = "SELECT * FROM aps2.aps3";
            log(self.file,"Data 3 extraction started")
            df3 = pd.DataFrame(list(session.execute(query)))
            df3=df3.sort_values(by=['ind'])
            df3.drop("ind",axis=1,inplace=True)
            
            log(self.file,"Data 3 extraction completed")

            df=pd.concat([df1,df2,df3],axis=1)
            df.to_csv("data/raw/aps.csv",index=False)


        except Exception as e:
            print(e)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    Data_extraction().get(config_path=parsed_args.config)