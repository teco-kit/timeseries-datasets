import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA

# ========================================       DSA_HAR_DATA               =============================
class DSA_HAR_DATA(BASE_DATA):
    """
    https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities

    Daily and Sports Activities Data Set

    Brief Description of the Dataset:
    ---------------------------------
    Each of the 19 activities is performed by eight subjects (4 female, 4 male, between the ages 20 and 30) for 5 minutes.
    Total signal duration is 5 minutes for each activity of each subject.
    The subjects are asked to perform the activities in their own style and were not restricted on how the activities should be performed. 
    For this reason, there are inter-subject variations in the speeds and amplitudes of some activities.
	
    The activities are performed at the Bilkent University Sports Hall, in the Electrical and Electronics Engineering Building, and in a flat outdoor area on campus. 
    Sensor units are calibrated to acquire data at 25 Hz sampling frequency. 
    The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity.
    """

    def __init__(self, args):
        super(DSA_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # there are total 3 sensors :ACC Gyro Mag
        # amounted in 5 places "T", "RA", "LA", "RL", "LL"
        # In total 45 Channels
		
        self.used_cols = list(np.arange(45))

        col_list    =  ["acc_x","acc_y","acc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
        pos_list = ["T", "RA", "LA", "RL", "LL"]
        self.col_names = [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]


        # TODO , here the keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3,4,5]

        self.vali_keys    = [6,7]

        self.test_keys    = [8]

        self.drop_activities = []

        self.file_encoding = {}  # no use 
        
        self.label_map = [(0, '01'), # sitting (A1),
                          (1, "02"), # standing (A2),
                          (2, "03"), # lying on back and on right side (A3 and A4),
                          (3, "04"), # lying on back and on right side (A3 and A4),
                          (4, "05"), # ascending and descending stairs (A5 and A6),
                          (5, "06"), # ascending and descending stairs (A5 and A6),
                          (6, "07"), # standing in an elevator still (A7)
                          (7, "08"), # and moving around in an elevator (A8),
                          (8, "09"), # walking in a parking lot (A9),
                          (9, "10"), # walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11),
                          (10, "11"), # walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11),
                          (11, "12"), # running on a treadmill with a speed of 8 km/h (A12),
                          (12, "13"), # exercising on a stepper (A13),
                          (13, "14"), # exercising on a cross trainer (A14),
                          (14, "15"), # cycling on an exercise bike in horizontal and vertical positions (A15 and A16),
                          (15, "16"), # cycling on an exercise bike in horizontal and vertical positions (A15 and A16),
                          (16, "17"), # rowing (A17),
                          (17, "18"), # jumping (A18),
                          (18, "19")] #  playing basketball (A19).

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()


    def load_the_data(self, root_path):

        temp_train_keys = []
        temp_test_keys = []
        temp_vali_keys = []

        df_dict = {}

        for action in os.listdir(root_path):
            action_name = action[1:]

            for user in os.listdir(os.path.join(root_path,action)):
                user_name = user[1:]

                for seg in os.listdir(os.path.join(root_path,action,user)):
                    seg_name = seg[1:3]

                    sub_data = pd.read_csv(os.path.join(root_path,action,user,seg),header=None)
                    sub_data =sub_data.iloc[:,self.used_cols]
                    sub_data.columns = self.col_names

                    sub_id = "{}_{}_{}".format(user_name,seg_name,action_name)
                    sub_data["sub_id"] = sub_id

                    sub_data["activity_id"] = action_name

                    # update the keys 
                    if int(user_name) in self.train_keys:
                        temp_train_keys.append(sub_id)
                    elif int(user_name) in self.vali_keys:
                        temp_vali_keys.append(sub_id)
                    else:
                        temp_test_keys.append(sub_id)

                    df_dict[sub_id] = sub_data

        self.train_keys   = temp_train_keys
        self.vali_keys    = temp_vali_keys
        self.test_keys    = temp_test_keys

        df_all = pd.concat(df_dict)

        label_mapping = {item[1]:item[0] for item in self.label_map}
        # because the activity label in the df is not encoded, thet are  "01","02",...,"19"
        # first, map them in to nummeric number
        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
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

