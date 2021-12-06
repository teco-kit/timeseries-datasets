import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer
# Note! the 0 class, some paper does not use it some dose!

# ========================================       Opportunity HAR UCI                =============================
class Opportunity_HAR_DATA():

    def __init__(self, args, flag="train"):
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


        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)
        # the sensors between 134 and 248 are amounted on devices, so they will not to be considered
        self.used_cols = [1,  2,   3, # Accelerometer RKN^ 
                          4,  5,   6, # Accelerometer HIP
                          7,  8,   9, # Accelerometer LUA^ 
                          10, 11,  12, # Accelerometer RUA_
                          13, 14,  15, # Accelerometer LH
                          16, 17,  18, # Accelerometer BACK
                          19, 20,  21, # Accelerometer RKN_ 
                          22, 23,  24, # Accelerometer RWR
                          25, 26,  27, # Accelerometer RUA^
                          28, 29,  30, # Accelerometer LUA_ 
                          31, 32,  33, # Accelerometer LWR
                          34, 35,  36, # Accelerometer RH
                          37, 38,  39, 40, 41, 42, 43, 44, 45, # InertialMeasurementUnit BACK
                          50, 51,  52, 53, 54, 55, 56, 57, 58, # InertialMeasurementUnit RUA
                          63, 64,  65, 66, 67, 68, 69, 70, 71, # InertialMeasurementUnit RLA 
                          76, 77,  78, 79, 80, 81, 82, 83, 84, # InertialMeasurementUnit LUA
                          89, 90,  91, 92, 93, 94, 95, 96, 97,  # InertialMeasurementUnit LLA
                          102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, # InertialMeasurementUnit L-SHOE
                          118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, # InertialMeasurementUnit R-SHOE
                          249  # Label
                         ]

        self.train_keys   = [               'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat',  'S1-ADL5.dat', 'S1-Drill.dat', # subject 1
                             'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                                'S2-Drill.dat', # subject 2
                             'S3-ADL1.dat', 'S3-ADL2.dat',                                                               # subject 3
                             'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat',                 'S4-ADL5.dat', 'S4-Drill.dat'] # subject 4


        self.vali_keys    = [ 'S1-ADL1.dat','S3-ADL3.dat', 'S3-Drill.dat','S4-ADL4.dat',]
        self.test_keys    = ['S2-ADL4.dat', 'S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat']

        self.drop_activities = []
        col_names         = ["dim_{}".format(i) for i in range(len(self.used_cols)-1)]
        self.col_names    =  col_names + ["activity_id"]
        
        self.label_map = [(0,      'Other'),
                          (406516, 'Open Door 1'),
                          (406517, 'Open Door 2'),
                          (404516, 'Close Door 1'),
                          (404517, 'Close Door 2'),
                          (406520, 'Open Fridge'),
                          (404520, 'Close Fridge'),
                          (406505, 'Open Dishwasher'),
                          (404505, 'Close Dishwasher'),
                          (406519, 'Open Drawer 1'),
                          (404519, 'Close Drawer 1'),
                          (406511, 'Open Drawer 2'),
                          (404511, 'Close Drawer 2'),
                          (406508, 'Open Drawer 3'),
                          (404508, 'Close Drawer 3'),
                          (408512, 'Clean Table'),
                          (407521, 'Drink from Cup'),
                          (405506, 'Toggle Switch')]

        self.file_encoding = {'S1-ADL1.dat':11, 'S1-ADL2.dat':12, 'S1-ADL3.dat':13, 'S1-ADL4.dat':14, 'S1-ADL5.dat':15, 'S1-Drill.dat':16,
                              'S2-ADL1.dat':21, 'S2-ADL2.dat':22, 'S2-ADL3.dat':23, 'S2-ADL4.dat':24, 'S2-ADL5.dat':25, 'S2-Drill.dat':26,
                              'S3-ADL1.dat':31, 'S3-ADL2.dat':32, 'S3-ADL3.dat':33, 'S3-ADL4.dat':34, 'S3-ADL5.dat':35, 'S3-Drill.dat':36,
                              'S4-ADL1.dat':41, 'S4-ADL2.dat':42, 'S4-ADL3.dat':43, 'S4-ADL4.dat':44, 'S4-ADL5.dat':45, 'S4-Drill.dat':46}

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        
        self.read_data()


    def read_data(self):

        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path     = self.root_path)

        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
            
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

        train_vali_window_index = self.get_the_sliding_index(train_vali_x.copy(), train_vali_y.copy())
        self.test_window_index = self.get_the_sliding_index(test_x.copy(), test_y.copy())
        #print("train_vali_window_index:",len(train_vali_window_index))
        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()
    def load_the_data(self, root_path):
        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if file[-3:]=="dat"] # in total , it should be 24
        assert len(file_list) == 24
        df_dict = {}

        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # TODO check missing labels? 
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            # label transformation
            sub_data["activity_id"] = sub_data["activity_id"].map(self.labelToId)
            sub_data['sub_id'] = self.file_encoding[file]
            df_dict[self.file_encoding[file]] = sub_data
            
        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            train_vali = pd.concat([train_vali,df_dict[self.file_encoding[key]]])

        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[self.file_encoding[key]]])
        
        # the col position varies between different datasets

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

    def get_the_sliding_index(self, data_x, data_y):
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
        freq         = 30
        windowsize   = int(1 * freq)
        displacement = int(0.5*freq)
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

                    window_index.append([sub_id, start, end])

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
            vali_keys    = [  self.file_encoding[key] for key in self.vali_keys]
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

