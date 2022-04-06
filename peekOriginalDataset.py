normal_data_path = "/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Normal_v1.csv"
attack_data_path = "/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Attack_v0.csv"
attackFeatureInfo_csv_path= "/workspace/lab/anomaly_detecton/dataset/SWAT/List_of_attacks_Final.csv"
import pandas as pd


def SWAT_loadData(data_path):
    ############################### Normal 
    normal = pd.read_csv(data_path)#, nrows=1000)
    normal["Timestamp"] = normal["Timestamp"].str.strip()
    normal["Timestamp"] = pd.to_datetime(normal["Timestamp"],format="%d/%m/%Y %I:%M:%S %p")
    for i in list(normal): 
        normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
    return normal

if __name__ == "__main__":
    df = SWAT_loadData(attack_data_path)
    with open("peekOriginalDataset_result.txt","w") as f:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print("df\n",df[df["P102"]=="2"][["Timestamp","P101","P102"]],file=f)




