import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer

# ========================================       Opportunity HAR UCI                =============================
class Daphnet_HAR_DATA():

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


        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)
        # Time of sample in millisecond
        # Ankle (shank) acceleration - horizontal forward acceleration [mg]
        # Ankle (shank) acceleration - vertical [mg]
        # Ankle (shank) acceleration - horizontal lateral [mg]
        # Upper leg (thigh) acceleration - horizontal forward acceleration [mg]
        # Upper leg (thigh) acceleration - vertical [mg]
        # Upper leg (thigh) acceleration - horizontal lateral [mg]
        # Trunk acceleration - horizontal forward acceleration [mg]
        # Trunk acceleration - vertical [mg]
        # Trunk acceleration - horizontal lateral [mg]
        # Annotations (see Annotations section)

        self.used_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.train_keys   = ['S01R01.txt', 'S01R02.txt',
                             #'S02R01.txt','S02R02.txt'
                             'S03R01.txt', 'S03R02.txt',# 'S03R03.txt', 
                             #'S04R01.txt',
                             # 'S05R01.txt','S05R02.txt'
                             'S06R01.txt', 'S06R02.txt',
                             'S07R01.txt', 'S07R02.txt',
                             'S08R01.txt', 
                             'S09R01.txt', 
                             'S10R01.txt' ]

        self.vali_keys    =['S02R02.txt', 
                            'S03R03.txt', 
                            'S05R01.txt' ]

        self.test_keys    = ['S02R01.txt',
                             'S04R01.txt',
                             'S05R02.txt']

        self.drop_activities = [0]

        self.col_names    =  ["acc_h_f_ankle", "acc_v_ankle", "acc_h_l_ankle",
                              "acc_h_f_leg", "acc_v_leg", "acc_h_l_leg",
                              "acc_h_f_trunk", "acc_v_trunk","acc_h_l,trunk",
                              "activity_id"]
        
        self.file_encoding = {'S01R01.txt':11, 'S01R02.txt':12,
                              'S02R01.txt':21, 'S02R02.txt':22,
                              'S03R01.txt':31, 'S03R02.txt':32, 'S03R03.txt':33,
                              'S04R01.txt':41,
                              'S05R01.txt':51, 'S05R02.txt':52,
                              'S06R01.txt':61, 'S06R02.txt':62,
                              'S07R01.txt':71, 'S07R02.txt':72,
                              'S08R01.txt':81,
                              'S09R01.txt':91,
                               'S10R01.txt':101}
        
        self.label_map = [
            (0, 'Other'),
            (1, 'No freeze'),
            (2, 'Freeze')
        ]



        
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
        file_list = os.listdir(root_path)

        assert len(file_list) == 17
        df_dict = {}

        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file),header=None, delim_whitespace=True)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # TODO check missing labels? 
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            # This data set dose not need label transformation

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
        freq         = 64  
        windowsize   = int(3 * freq)

        if Flag_id:
            displacement = int(0.33 * freq)
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

