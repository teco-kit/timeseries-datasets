import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer

# ========================================       Data loader Base class               =============================
class BASE_DATA():

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

        # the following parameters are all for slidingwindow set up 
        self.freq         = args.sampling_freq  
        self.windowsize   = args.windowsize
        self.displacement_train = args.displacement_train
        self.displacement_test = args.displacement_test
        self.drop_long    = args.drop_long


        # Data Set spezifisch parameters
        # self.used_cols = []
		# self.train_keys   = []
        # self.vali_keys    =[]
        # self.test_keys    = []
        # self.drop_activities = []
        # self.col_names    =  []
        # self.file_encoding = {}
        # self.label_map = []


    def read_data(self):
        # Load the Data, Train Test Split
        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path     = self.root_path)

        # Data Pre Processing : Differencing
        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
        # Data Pre Processing : Normalization         
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

        # Get the Siding Window indexes!
        train_vali_window_index = self.get_the_sliding_index(train_vali_x.copy(), train_vali_y.copy(), Flag_id = True)
        self.test_window_index = self.get_the_sliding_index(test_x.copy(), test_y.copy(), Flag_id = False)

        # Train Vali Split
        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        # Save the Data
        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()

    def load_the_data(self, root_path):
        raise NotImplementedError


    def differencing(self, train_vali, test):
        # define the name for differenced columns
        columns = ["diff_"+i for i in train_vali.columns]

        # The original data has been divided into segments by sub_id: a segment belongs to a same user 
        # There is no continuity between different segments, so diffrecne is only done within each segment

        # Train_vali_diff
        diff_train_vali = []
        for id in train_vali.index.unique():
            diff_train_vali.append(train_vali.loc[id].diff())

        diff_train_vali = pd.concat(diff_train_vali)
        diff_train_vali.columns = columns
        diff_train_vali.fillna(method ="backfill",inplace=True)
        train_vali = pd.concat([train_vali,diff_train_vali], axis=1)

        # Test_diff
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


        freq         = self.freq   
        windowsize   = self.windowsize

        if Flag_id:
            displacement = self.displacement_train
        else:
            displacement = self.displacement_test

        drop_long    = self.drop_long 
        drop_ubergang = False

        if drop_ubergang == True:
            drop_index = []
            numblocks = data_x['act_block'].max()
            for block in range(1, numblocks+1):
                drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
                drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
            data_x = data_x.drop(drop_index)

        window_index = []
        for index in data_x.act_block.unique():

            temp_df = data_x[data_x["act_block"]==index]
            assert len(temp_df["activity_id"].unique())==1

            if temp_df["activity_id"].unique()[0] not in self.drop_activities:
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]
                end   = start+windowsize

                while end <= temp_df.index[-1] + 1:
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

