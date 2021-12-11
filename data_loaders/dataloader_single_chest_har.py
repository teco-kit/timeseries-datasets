import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA

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
        super(Single_Chest_HAR_DATA, self).__init__(args)
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

        # TODO  There are total 15 users
        # TODO , here the keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3, 6,7,8, 12,13,14]
        self.vali_keys    = [4,9]
        self.test_keys    = [5,10,11,15]

        self.drop_activities = [0]

        self.file_encoding = {}  # no use 
        
        self.label_map = [(0, 'other'), 
                          (1, 'Working at Computer'),
                          (2, 'Standing Up, Walking and Going up\down stairs'),
                          (3, 'Standing'), 
                          (4, 'Walking'),
                          (5, 'Going Up\Down Stairs'),
                          (6, "Walking and Talking with Someone"),
                          (7, "Talking while Standing")]

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

    def load_the_data(self, root_path):

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if file.endswith('.csv')]

        df_dict = {}
        for file in file_list:

            sub_data = pd.read_csv(os.path.join(root_path,file), header=None)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # if missing values, imputation
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')

            sub_id = int(file.split(".")[0])
            sub_data['sub_id'] = sub_id

            df_dict[sub_id] = sub_data


        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            train_vali = pd.concat([train_vali,df_dict[key]])
        # the label is alweady encoded, here is not necessary
        train_vali["activity_id"] = train_vali["activity_id"].map(self.labelToId)


        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[key]])
        # the label is alweady encoded, here is not necessary
        test["activity_id"] = test["activity_id"].map(self.labelToId)

        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label
