import pandas as pd
import numpy as np
import os

import scipy.io as sio
from data_loaders.dataloader_base import BASE_DATA

# ================================= USC_HAD_HAR_DATA ============================================
class USC_HAD_HAR_DATA(BASE_DATA):
    """

    **********************************************
    Section 1: Device Configuration


    2. Sampling rate: 100Hz
    3. Accelerometer range: +-6g
    4. Gyroscope range: +-500dps


    **********************************************
    Section 2: Data Format
    Each activity trial is stored in an .mat file.

    The naming convention of each .mat file is defined as:
    a"m"t"n".mat, where
    "a" stands for activity
    "m" stands for activity number
    "t" stands for trial
    "n" stands for trial number

    Each .mat file contains 13 fields:
    1. title: USC Human Motion Database
    2. version: it is version 1.0 for this first round data collection
    3. date
    4. subject number
    5. age
    6. height
    7. weight
    8. activity name
    9. activity number
    10. trial number
    11. sensor_location
    12. sensor_orientation
    13. sensor_readings

    For sensor_readings field, it consists of 6 readings:
    From left to right:
    1. acc_x, w/ unit g (gravity)
    2. acc_y, w/ unit g
    3. acc_z, w/ unit g
    4. gyro_x, w/ unit dps (degrees per second)
    5. gyro_y, w/ unit dps
    6. gyro_z, w/ unit dps

    **********************************************
    Section 3: Activities
    1. Walking Forward
    2. Walking Left
    3. Walking Right
    4. Walking Upstairs
    5. Walking Downstairs
    6. Running Forward
    7. Jumping Up
    8. Sitting
    9. Standing
    10. Sleeping
    11. Elevator Up
    12. Elevator Down

    """

    def __init__(self, args):
        super(USC_HAD_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # !!!!!! Depending on the setting of each data set!!!!!!
        # because this dataset only has 6 columns, the label is saved in the file name, so this used cols will not be used
        self.used_cols    = [0,1,2,3,4,5]
        # This dataset only has this 6 channels
        self.col_names = [ 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z' ]

        # The original labels are from 1 to 12
        self.label_map = [(1, "Walking Forward"),
                          (2, "Walking Left"),
                          (3, "Walking Right"),
                          (4, "Walking Upstairs"),
                          (5, "Walking Downstairs"),
                          (6, "Running Forward"),
                          (7, "Jumping Up"),
                          (8, "Sitting"),
                          (9, "Standing"),
                          (10, "Sleeping"),
                          (11, "Elevator Up"),
                          (12, "Elevator Down")]

        # As can be seen from the readme
      
        self.drop_activities = []
        self.file_encoding = {}  # no use 

        self.train_keys   = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

        self.vali_keys    = [ 11, 12 ]

        self.test_keys    = [ 13, 14 ]   
	
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

    def load_the_data(self, root_path):

        activities = range(1, 13)


        temp_train_keys = []
        temp_vali_keys = []
        temp_test_keys = []

        df_dict = {}
        for subject in range(1, 15):

            for activity in activities:

                for trial in range(1, 6):

                    sub_data = sio.loadmat("%s/Subject%d%sa%dt%d.mat" % (root_path, subject, os.sep, activity, trial))
                    sub_data = pd.DataFrame(np.array(sub_data['sensor_readings']))

                    sub_data =sub_data.iloc[:,self.used_cols]
                    sub_data.columns = self.col_names
					
                    id_ = "{}_{}_{}".format(subject,activity,trial)
                    sub_data["sub_id"] = id_

                    sub_data["activity_id"] = activity
                    df_dict[id_] = sub_data   

                    if subject in self.train_keys:
                        temp_train_keys.append(id_)
                    elif subject in self.vali_keys:
                        temp_vali_keys.append(id_)
                    else:
                        temp_test_keys.append(id_)

        self.train_keys = temp_train_keys 
        self.vali_keys = temp_vali_keys 
        self.test_keys = temp_test_keys 

        df_all = pd.concat(df_dict)
        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # train_vali Test split 
        train_vali = df_all.loc[self.train_keys+self.vali_keys]
        test = df_all.loc[self.test_keys]
        
        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1] 
        train_vali = train_vali.iloc[:,:-1]


        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label
