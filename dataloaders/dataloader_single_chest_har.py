import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       Single_Chest_HAR_DATA               =============================
class Single_Chest_HAR_DATA(BASE_DATA):

    """

    https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer

    Activity Recognition from Single Chest-Mounted Accelerometer Data Set

    BASIC INFO ABOUT THE DATA:
    ---------------------------------

    --- The dataset collects data from a wearable accelerometer mounted on the chest
    --- Sampling frequency of the accelerometer: 52 Hz
    --- Accelerometer Data are Uncalibrated
    --- Number of Participants: 15
    --- Number of Activities: 7
    --- Data Format: CSV
    """

    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # sequential number, x acceleration, y acceleration, z acceleration, label 
        self.used_cols = [1,2,3,4]
        # there are total 1 sensor with 3 channels
        self.col_names = ["acc_x","acc_y","acc_z","activity_id"]

        self.label_map = [(0, 'other'), 
                          (1, 'Working at Computer'),
                          (2, 'Standing Up, Walking and Going up\down stairs'),
                          (3, 'Standing'), 
                          (4, 'Walking'),
                          (5, 'Going Up\Down Stairs'),
                          (6, "Walking and Talking with Someone"),
                          (7, "Talking while Standing")]

        self.drop_activities = [0]

        # TODO  There are total 15 users
        # TODO , here the keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3,4, 6,7,8,9, 12,13,14]
        self.vali_keys    = []
        self.test_keys    = [5,10,11,15]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(Single_Chest_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if file.endswith('.csv')]

        assert len(file_list) == 15

        df_dict = {}
        for file in file_list:

            sub_data = pd.read_csv(os.path.join(root_path,file), header=None)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # if missing values, imputation
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')

            sub_id = int(file.split(".")[0])
            sub_data['sub_id'] = sub_id
            sub_data["sub"] = sub_id

            if sub_id not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub_id] = []
            self.sub_ids_of_each_sub[sub_id].append(sub_id)  

            df_dict[sub_id] = sub_data

        # all data
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        # label transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]
		
        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y