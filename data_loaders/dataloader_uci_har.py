import pandas as pd
import numpy as np
import os
from random import sample
from data_loaders.utils import Normalizer
# https://github.com/Koutoulakis/Deep-Learning-for-Human-Activity-Recognition/blob/master/Datareader/datareader.py
# https://github.com/nhammerla/deepHAR/blob/master/data/datareader.py
# https://github.com/rmutegeki/iSPLInception/blob/ca45393b8ff1cf1b57151893e7ffd65bdb1ecfb3/datareader.py#L358
# https://github.com/AntonioAcunzo/mvts_transformer/tree/master/src/datasets
# ============================== UCI HAR  ======================================
class UCI_HAR_DATA():

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
        
        # All ID used for training [ 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
        # All ID used for Test  [ 2,  4,  9, 10, 12, 13, 18, 20, 24]
		
        # If train_subs is the same as all the IDs used for training, then split randomly
        # If train_subs is only subset of all ID used for training, the rest IDs (subjects) will be used for validation
        self.train_subs = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17,19, 21, 22, 23, 25, 26, 27, 28, 29, 30 ]
        

        self.spectrogram  = args.spectrogram
        if self.spectrogram:
            self.scales       = np.arange(1, int(args.f_max/2)+1)
            self.wavelet      = args.wavelet

        self.read_data()

    def read_data(self):
        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path = self.root_path)
        
        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
            
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)
            
            
        train_x,train_y,vali_x,vali_y = self.train_vali_split(train_vali_x,train_vali_y)
        
        self.train_x = train_x.copy()
        # !!!!!! Note here that different datasets are labelled differently
        # UCIHAR are [1,2,3,4,5,6], --> so all minus one -- [0,1,2,3,4,5]
        self.train_y = np.array(train_y.copy())-1
        self.vali_x = vali_x.copy()
        self.vali_y = np.array(vali_y.copy())-1
        self.test_x = test_x.copy()
        self.test_y = test_y.copy().iloc[:,0].values-1

    def load_the_data(self, root_path):
        # For each dataset, How to read data is different
        train_vali_path = os.path.join(root_path, "train/Inertial Signals/")
        test_path  = os.path.join(root_path, "test/Inertial Signals/")
        
        file_list = os.listdir(train_vali_path)

        train_vali_dict = {}
        test_dict  = {}
        for file in file_list:
            train_vali = pd.read_csv(train_vali_path + file,header=None, delim_whitespace=True)
            test  = pd.read_csv(test_path+file[:-9]+"test.txt",header=None, delim_whitespace=True)
            train_vali_dict[file[:-9]] = train_vali
            test_dict[file[:-9]] = test
    
        columns = ['body_acc_x_', 'body_acc_y_', 'body_acc_z_',
                   'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
                   'total_acc_x_', 'total_acc_y_', 'total_acc_z_']


        # Train_Vali: Convert all channels into one DataFrame, and Prepare the index for each segment!
        train_vali = pd.DataFrame(np.stack([train_vali_dict[col].values.reshape(-1) for col in columns], axis=1), columns = columns)
        index = []
        for i in range(train_vali_dict["body_acc_x_"].shape[0]):
            index.extend(128*[i])
        train_vali.index = index

        # Test: Convert all channels into one DataFrame, and Prepare the index for each segment!
        test = pd.DataFrame(np.stack([test_dict[col].values.reshape(-1) for col in columns], axis=1), columns = columns)
        index = []
        for i in range(test_dict["body_acc_x_"].shape[0]):
            index.extend(128*[i])
        test.index = index

        # read the label
        train_vali_label = pd.read_csv(os.path.join(root_path,"train/y_train.txt"),header=None)
        test_label = pd.read_csv(os.path.join(root_path,"test/y_test.txt"),header=None)
        return train_vali, train_vali_label, test, test_label
        
    def differencing(self, train_vali, test):
        # define the name for differenced columns
        columns = ["diff_"+i for i in train_vali.columns]
        # The original data has been divided into segments by the sliding window method. 
        # There is no continuity between paragraphs, so diffrecne is only done within each segment
        grouped_train_vali = train_vali.groupby(by=train_vali.index)
        diff_train_vali = grouped_train_vali.diff()
        diff_train_vali.columns = columns
		# TODO backfill?forefill?
        diff_train_vali.fillna(method ="backfill",inplace=True)
        train_vali = pd.concat([train_vali,diff_train_vali], axis=1)

        grouped_test = test.groupby(by=test.index)
        diff_test = grouped_test.diff()
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

    def train_vali_split(self, train_vali, train_vali_label) :
        # This file provides information about which ID each segment belongs to
        # It is for train test split according to the IDs
        train_vali_subjects = pd.read_csv(os.path.join(self.root_path,"train/subject_train.txt"), header=None)
        train_vali_subjects.columns = ["subjects"]
        
        all_ids = set(train_vali_subjects["subjects"].unique())
        train_subs = set(self.train_subs)
        
        train_label = []
        train_dfs = []
        vali_label = []
        vali_dfs = []
        
        if len(all_ids.difference(train_subs)) > 0:
            print("subjects split")

            for i in range(train_vali_subjects.shape[0]):
                temp = train_vali.loc[i]
                temp_label = train_vali_label.loc[i].values[0]
                id = train_vali_subjects.loc[i].values[0]
                if id in train_subs:
                    train_dfs.append(temp)
                    train_label.append(temp_label)
                else:
                    vali_dfs.append(temp)
                    vali_label.append(temp_label)
        else:
            print("data ramdom split")
            all_index = list(train_vali.index.unique())
            train_list = sample(all_index,int(len(all_index)*0.8))
            for i in all_index:
                temp = train_vali.loc[i]
                temp_label = train_vali_label.loc[i].values[0]
                id = train_vali_subjects.loc[i].values[0]
                if id in train_list:
                    train_dfs.append(temp)
                    train_label.append(temp_label)
                else:
                    vali_dfs.append(temp)
                    vali_label.append(temp_label)
            
        train = pd.concat([train_dfs[i].reset_index(drop=True).set_index(pd.Series(128*[i])) for i in range(len(train_dfs))], axis=0)
        vali = pd.concat([vali_dfs[i].reset_index(drop=True).set_index(pd.Series(128*[i])) for i in range(len(vali_dfs))], axis=0)    
        return train,train_label,vali, vali_label
