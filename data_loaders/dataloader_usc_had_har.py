import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer
import scipy.io as sio

# https://github.com/saif-mahmud/self-attention-HAR/blob/main/configs/data.yaml
# https://github.com/esansano/dl-for-har-comparison/blob/master/src/utils/dataimport.py

# ================================= PAMAP2 HAR DATASET ============================================
class USC_HAD_HAR_DATA():

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
		
        # !!!!!! Depending on the setting of each data set!!!!!!
        # because this dataset only has 6 columns, the label is saved in the file name, so this used cols will not be used
        self.used_cols    = []
        # The original labels are from 1 to 12, here we substract all label - 1 !!!!
        self.label_map = {0: "Walking Forward",
                          1: "Walking Left",
                          2: "Walking Right",
                          3: "Walking Upstairs",
                          4: "Walking Downstairs",
                          5: "Running Forward",
                          6: "Jumping Up",
                          7: "Sitting",
                          8: "Standing",
                          9: "Sleeping",
                          10: "Elevator Up",
                          11: "Elevator Down"}

        # As can be seen from the readme
      
        self.drop_activities = []

        self.train_keys   = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

        self.vali_keys    = [ 11, 12 ]

        self.test_keys    = [ 13, 14 ]   
	
        # This dataset only has this 6 channels
        self.col_names = [ 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z' ]

        self.read_data()
        
    def read_data(self):

        # load the data
        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path = self.root_path)

        # differencing 
        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
        # normalization
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)
        # sliding window
        train_vali_window_index = self.get_the_sliding_index(train_vali_x.copy(), train_vali_y.copy(),Flag_id = True)
        self.test_window_index = self.get_the_sliding_index(test_x.copy(), test_y.copy(),Flag_id = False)

        # train validation split
        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()

    def load_the_data(self, root_path):

        activities = range(1, 13)


        temp_train_keys = []
        temp_vali_keys = []
        temp_test_keys = []

        df_dict = {}
        for subject in range(1, 15):
            for activity in activities:
                for trial in range(1, 6):

                    data = sio.loadmat("%s/Subject%d%sa%dt%d.mat" % (root_path, subject, os.sep, activity, trial))
                    data = np.array(data['sensor_readings'])#[::2]  # Only even rows -> sampling rate 50Hz
                    data = pd.DataFrame(data,columns=self.col_names)
					
                    id_ = "{}_{}_{}".format(subject,activity,trial)
                    data["sub_id"] = id_
                    if subject in self.train_keys:
                        temp_train_keys.append(id_)
                    elif subject in self.vali_keys:
                        temp_vali_keys.append(id_)
                    else:
                        temp_test_keys.append(id_)

                    data["activity_id"] = activity
                    df_dict[id_] = data   

        self.train_keys = temp_train_keys 
        self.vali_keys = temp_vali_keys 
        self.test_keys = temp_test_keys 



        train_vali = pd.DataFrame()
        for key in self.train_keys+self.vali_keys :
            train_vali = pd.concat([train_vali,df_dict[key]])

        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[key]])
        
        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1] - 1
        train_vali = train_vali.iloc[:,:-1]


        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1] - 1
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
        freq         = 100
        windowsize   = int(1 * freq)
        if Flag_id:
            displacement = int(0.4 * freq)
        else:
            displacement = 1
        drop_long    = 7.5
        window_index = []
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