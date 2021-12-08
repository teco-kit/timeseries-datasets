import pandas as pd
import numpy as np
import os
from random import sample
import scipy.io as sio
from data_loaders.utils import Normalizer
# ========================================       Skoda_HAR_DATA               =============================
class Skoda_HAR_DATA():

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


        # Column 1: label
        # Column 2+s*7: sensor id
        # Column 2+s*7+1: X acceleration calibrated
        # Column 2+s*7+2: Y acceleration calibrated
        # Column 2+s*7+3: Z acceleration calibrated
        # Column 2+s*7+4: X acceleration raw
        # Column 2+s*7+5: Y acceleration raw
        # Column 2+s*7+6: Z acceleration raw

        self.used_cols = [0]+[2 + s * 7 for s in range(10)] + [3 + s *7 for s in range(10)] + [4 + s *7 for s in range(10)]
        self.used_cols.sort()

        self.train_keys   = []  # no use 

        self.vali_keys    = []  # no use 

        self.test_keys    = []  # no use 

        self.drop_activities = [32]

        # there are total 30 sensors 
        col_names = ["acc_x","acc_y", "acc_z"]
        self.col_names    =  ["activity_id"] + [j  for k in [[item+"_"+str(i) for item in col_names] for i in range(1,11)] for j in k ]
        
        self.file_encoding = {}  # no use 
        
        self.label_map = [(32, "null class"),
                          (48, "write on notepad"),
                          (49, "open hood"),
                          (50, "close hood"),
                          (51, "check gaps on the front door"),
                          (52, "open left front door"),
                          (53, "close left front door"),
                          (54, "close both left door"),
                          (55, "check trunk gaps"),
                          (56, "open and close trunk"),
                          (57, "check steering wheel")]

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



        data_dict = sio.loadmat(file_name=os.path.join(root_path,"right_classall_clean.mat"), squeeze_me=True)
        data = data_dict[list(data_dict.keys())[3]]

        data = data[:, self.used_cols]
        data = pd.DataFrame(data,columns=self.col_names)

        data["sub_id"] = 1
        data["activity_id"]=data["activity_id"].map(self.labelToId)

        # Train Test Split
        data['act_block'] = ((data['activity_id'].shift(1) != data['activity_id']) | (data['sub_id'].shift(1) != data['sub_id'])).astype(int).cumsum()

        train_vali = pd.DataFrame()
        test = pd.DataFrame()

        for ac_id in data['activity_id'].unique():
            if ac_id not in self.drop_activities:
                temp_df = data[data["activity_id"]==ac_id]
                act_block_list = list(temp_df["act_block"].unique())
                # take the first 80% as the training set, and the rest 20% as the test set 
                train_vali_act_block = act_block_list[:int(0.8*len(act_block_list))]
                # test_act_blocks = act_block_list[int(0.8*len(act_block_list)):]

                for act_block in temp_df["act_block"].unique():
                    if act_block in train_vali_act_block:
                        train_vali = pd.concat([train_vali,temp_df[temp_df["act_block"]==act_block]])
                    else:
                        test = pd.concat([test,temp_df[temp_df["act_block"]==act_block]])


        del train_vali["act_block"]
        del test["act_block"]

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
        freq         = 98  
        windowsize   = int(1 * freq)

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

