import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer
# Note! the 0 class, some paper does not use it some dose!

# ========================================       FUSION_HAR_DATA             =============================
class FUSION_HAR_DATA():

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


        # There is only one file which includs the collected data from 33 users
        # delete the second column it is the timestamp
        self.used_cols = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, # Left_pocket
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, # Right_pocket
                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, # Wrist
                         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # Upper_arm
                         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Belt
                         69]
        # TODO This should be referenced by other paper
        self.train_keys   = ["Participant_1.csv", "Participant_2.csv", "Participant_3.csv",
                             "Participant_4.csv", "Participant_5.csv", "Participant_6.csv"]


        self.vali_keys    = ["Participant_7.csv", "Participant_8.csv"]
        self.test_keys    = ["Participant_9.csv", "Participant_10.csv"]

        self.drop_activities = []

        col_list    =  ["acc_x","acc_y","acc_z","lacc_x","lacc_y","lacc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
        pos_list = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
        self.col_names    =  [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]+["activity_id"]
        
        self.label_map = [(0, 'walking'), 
                          (1, 'standing'),
                          (2, 'jogging'),
                          (3, 'sitting'), 
                          (4, 'biking'),
                          (5, 'upstairs'),
                          (6, "downstairs")]

        self.file_encoding = {"Participant_1.csv":1, "Participant_2.csv":2, "Participant_3.csv":3,
                              "Participant_4.csv":4, "Participant_5.csv":5, "Participant_6.csv":6,
                              "Participant_7.csv":7, "Participant_8.csv":8, "Participant_9.csv":9,
                              "Participant_10.csv":10} 


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

        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()
    def load_the_data(self, root_path):

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if "Participant" in file] # in total , it should be 10
        assert len(file_list) == 10

        df_dict = {}

        for file in file_list:
            sub_data = pd.read_csv(os.path.join(root_path, file), skiprows=[1], header=None).drop(index=[0])
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # if missing values, imputation TODO
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub_data['sub_id'] =int(self.file_encoding[file])
            df_dict[self.file_encoding[file]] = sub_data

        label_mapping = {'walking':0, 
                         'standing':1,
                          'jogging':2,
                          'sitting':3, 
                          'biking':4,
                          'upstairs':5,
                         "downstairs":6}

        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            train_vali = pd.concat([train_vali,df_dict[self.file_encoding[key]]])

        def change_the_typo(label):
            if label == "upsatirs":
                return "upstairs"
            else:
                return label
        train_vali["activity_id"] = train_vali["activity_id"].apply(change_the_typo)
        train_vali["activity_id"] = train_vali["activity_id"].map(label_mapping)
        train_vali["activity_id"] = train_vali["activity_id"].map(self.labelToId)


        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[self.file_encoding[key]]])

        test["activity_id"] = test["activity_id"].apply(change_the_typo)
        test["activity_id"] = test["activity_id"].map(label_mapping)
        test["activity_id"] = test["activity_id"].map(self.labelToId)

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
        freq         = 20
        windowsize   = int(1.5 * freq)

        if Flag_id:
            displacement = int(0.5 * freq)
        else:
            displacement = 1

        drop_long    = 1
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

                while end < temp_df.index[-1]:
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

