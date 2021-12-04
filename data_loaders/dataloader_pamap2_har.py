import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer
from sklearn.preprocessing import LabelEncoder
# ================================= PAMAP2 HAR DATASET ============================================
class PAMAP2_HAR_DATA():

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
		
        # !!!!!! Depending on the setting of each data set!!!!!!
        # the 0th column is time step 
        self.used_cols    = [1,# this is "label"
                             # TODO check the settings of other paper 
                             # the second column is heart rate (bpm) --> ignore?
                             # each IMU sensory has 17 channals , 3-19,20-36,38-53
                             # the first temp ignores
                             # the last four channel according to the readme are invalide
                             4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,        # IMU Hand
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
                             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49   # IMU ankle
                            ]

        self.label_map = [ # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        # As can be seen from the PerformedActivitiesSummary.pdf, some activities are not performed
        # TODO this should be chosen by reading related work
        # self.drop_activities = [0,9,10,11,18,19,20] #TODO check!!!!
        self.drop_activities = [0]

        self.train_keys   = ['subject101', 'subject102', 'subject103', 
                             'subject104', 
                             'subject107', 'subject108', 'subject109']
        # Manually encoding the subject 
        self.train_keys   = [1,2,3,4,7,8,9]

        self.vali_keys    = ['subject105']
        self.vali_keys    = [5]

        self.test_keys    = ['subject106']
        self.test_keys    = [6]        
	
        # form the columns name , [label, 12*[hand], 12*[chest], 12*[ankle]]
        col_names=['activity_id']

        IMU_locations = ['hand', 'chest', 'ankle']
        IMU_data      = ['acc_16_01', 'acc_16_02', 'acc_16_03',
                         'acc_06_01', 'acc_06_02', 'acc_06_03',
                         'gyr_01', 'gyr_02', 'gyr_03',
                         'mag_01', 'mag_02', 'mag_03']

        self.col_names = col_names + [item for sublist in [[dat+'_'+loc for dat in IMU_data] for loc in IMU_locations] for item in sublist]

        self.read_data()
        
    def read_data(self):


        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path = self.root_path)
        train_vali_y, test_y, self.drop_activities = self.transform_labels(train_vali_y, test_y,self.drop_activities)

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
        
        df_dict = {}
        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # if missing values, imputation
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub_data['sub_id'] =int(file[9])
            df_dict[int(file[9])] = sub_data   

        train_vali = pd.DataFrame()
        for key in self.train_keys+self.vali_keys :
            train_vali = pd.concat([train_vali,df_dict[key]])

        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[key]])
        
        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,0]
        train_vali = train_vali.iloc[:,1:]


        test = test.set_index('sub_id')
        test_label = test.iloc[:,0]  
        test = test.iloc[:,1:]

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
        freq         = 100
        windowsize   = int(5.12 * freq)
        displacement = 1*freq
        drop_long    = 7.5
        window_index = []
        id_          = 0
        drop_ubergang = False

        if drop_ubergang == True:
            drop_index = []
            numblocks = data_x['act_block'].max()
            for block in range(1, numblocks+1):
                drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
                drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
            data_x = data_x.drop(drop_index)

        #print(data_x.act_block.unique())
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


    def transform_labels(self, y_train, y_test, drop_activities = [0]):
        """
        Transform label to min equal zero and continuous
        For example if we have [1,3,4] --->  [0,1,2]
        Remenber: convert the the label for drop activities, it may change
        """
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        #print(np.unique(y_train_test))
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        to_drop = encoder.transform(drop_activities)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        
        new_y_train = pd.Series(new_y_train)
        new_y_train.index = y_train.index
        new_y_train.name = "activity_id"
        
        new_y_test =  pd.Series(new_y_test)
        new_y_test.index = y_test.index
        new_y_test.name = "activity_id"

        return new_y_train, new_y_test, to_drop

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
            for item in train_vali_window_index:
                sub_id = item[0]
                if sub_id in self.vali_keys:
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