import pandas as pd
import numpy as np
import os
from random import sample
import scipy.io as sio
from data_loaders.utils import Normalizer

# https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities

# Daily and Sports Activities Data Set

# ========================================       DSA_HAR_DATA               =============================
class DSA_HAR_DATA():

    def __init__(self, args):
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """
        self.root_path    = args.root_path

        self.difference   = args.difference
        self.datanorm_type= args.datanorm_type




        self.used_cols = list(np.arange(45))

        # there are total 45 sensors 
        col_list    =  ["acc_x","acc_y","acc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
        pos_list = ["T", "RA", "LA", "RL", "LL"]
        self.col_names = [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]


        # TODO 
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


    def read_data(self):

        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path     = self.root_path)

        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
            
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

        train_vali_window_index = self.get_the_sliding_index(train_vali_x.copy(), train_vali_y.copy(), Flag_id = True)
        self.test_window_index = self.get_the_sliding_index(test_x.copy(), test_y.copy(), Flag_id = False)
        #print("train_vali_window_index:",len(train_vali_window_index))
        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()





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

        label_mapping = {item[1]:item[0] for item in self.label_map}

        df_all = pd.concat(df_dict)

        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        train_vali = df_all.loc[self.train_keys+self.vali_keys]
        test = df_all.loc[self.test_keys]

        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label


    def differencing(self, train_vali, test):
        # define the name for differenced columns
        columns = ["diff_"+i for i in train_vali.columns]
        # The original data has been divided into segments by the sliding window method. 
        # There is no continuity between paragraphs, so diffrecne is only done within each segment

        # Train_vali_diff

        diff_train_vali = []
        for id in train_vali.index.unique():
            diff_train_vali.append(train_vali.loc[id].diff())
        diff_train_vali = pd.concat(diff_train_vali)
        diff_train_vali.columns = columns
        diff_train_vali.fillna(method ="backfill",inplace=True)
        train_vali = pd.concat([train_vali,diff_train_vali], axis=1)


        diff_test = []
        for id in test.index.unique():
            diff_test.append(test.loc[id].diff())
        diff_test = pd.concat(diff_test)
        diff_test.columns = columns
        diff_test.fillna(method ="backfill",inplace=True)
        test  = pd.concat([test, diff_test],  axis=1)

        return train_vali, test

    def normalization(self, train_vali, test):
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali)
        train_vali = self.normalizer.normalize(train_vali)
        test  = self.normalizer.normalize(test)
        return train_vali, test

    def get_the_sliding_index(self, data_x, data_y, Flag_id = True):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """

        data_x = data_x.reset_index()
        data_y = data_y.reset_index()

        data_x["activity_id"] = data_y["activity_id"]
        data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()

        # TODO s!!!!!!!!!!!!!!!   Dataset Dependent!!!!!!!!!!!!!!!!!!!! 
        # To set the window size and Sliding step
        freq         = 25  
        windowsize   = int(5 * freq)

        if Flag_id:
            displacement = int(0.5 * freq)
        else:
            displacement = 1

        drop_long    = 3
        window_index = []
        drop_ubergang = False

        if drop_ubergang == True:
            drop_index = []
            numblocks = data_x['act_block'].max()
            for block in range(1, numblocks+1):
                drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
                drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
            data_x = data_x.drop(drop_index)

        for index in data_x.act_block.unique():
            temp_df = data_x[data_x["act_block"]==index]
            if temp_df["activity_id"].unique()[0] not in self.drop_activities:
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]
                end   = start+windowsize

                while end <= temp_df.index[-1]+1:
                    if Flag_id:
                        window_index.append([sub_id, start, end])
                    else:
                        window_index.append([start, end])

                    start = start + displacement
                    end   = start + windowsize

        return window_index

    def train_vali_split(self, train_vali_window_index):
        """
        if vali_keys is not None ----> subject split 
        if vali_keys is None ------> random 80% split
        After train test split : the window index consists of only TWO components : start_index and end_index
        """
        train_window_index = []
        vali_window_index  = []
        if len(self.vali_keys):
            print("Subjects Split")
            if len(self.file_encoding)>0:
                vali_keys    = [self.file_encoding[key] for key in self.vali_keys]
            else:
                vali_keys = self.vali_keys

            for item in train_vali_window_index:
                sub_id = item[0]
                if sub_id in vali_keys:
                    vali_window_index.append([item[1],item[2]])
                else:
                    train_window_index.append([item[1],item[2]])
        else:
            print("Random split")
            all_index = list(np.arange(len(train_vali_window_index)))
            train_list = sample(all_index,int(len(all_index)*0.8))
            for i in all_index:
                item = train_vali_window_index[i]
                if i in train_list:
                    train_window_index.append([item[1],item[2]])
                else:
                    vali_window_index.append([item[1],item[2]])

        return train_window_index, vali_window_index

