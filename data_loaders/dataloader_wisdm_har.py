import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA

# ========================================       WISDM_HAR_DATA             =============================
class WISDM_HAR_DATA(BASE_DATA):

    """

    https://www.cis.fordham.edu/wisdm/dataset.php
    Wireless Sensor Data Mining (WISDM) Lab

    BASIC INFO ABOUT THE DATA:
    ---------------------------------

    Sampling rate:  20Hz (1 sample every 50ms)

    raw.txt follows this format: [user],[activity],[timestamp],[x-acceleration],[y-accel],[z-accel];

    Fields: *user  nominal, 1..36

    activity nominal, { Walking Jogging Sitting Standing Upstairs Downstairs }
    """

    def __init__(self, args):
        super(WISDM_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """


        # There is only one file which includs the collected data from 33 users
        # delete the second column it is the timestamp
        self.used_cols = [0,1,3,4,5]
        self.col_names    =  ['sub_id','activity_id','timestamp', 'acc_x', 'acc_y', 'acc_z']

        # TODO This should be referenced by other paper
        # TODO , here the keys for each set will be updated in the readtheload function

        self.train_keys   = [1,2,3,4,  7,8,9,10,  13,14,15,16,  19,20,21,22,  25,26,27,28,  31,32]
        self.vali_keys    = [5,11,17,23,29]
        self.test_keys    = [6,12,18,24,30,33]

        self.drop_activities = []


        self.label_map = [(0, 'Walking'), 
                          (1, 'Jogging'),
                          (2, 'Sitting'),
                          (3, 'Standing'), 
                          (4, 'Upstairs'),
                          (5, 'Downstairs')]

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()


    def load_the_data(self, root_path):

        df = pd.read_csv(os.path.join(root_path,"WISDM_ar_v1.1_raw.txt"),header=None,names=self.col_names)
        df["acc_z"]=df["acc_z"].replace('\;','',regex=True).astype(float) #清洗掉z-axis中的符号

        df =df.iloc[:,self.used_cols]

        df.dropna(inplace=True)

        label_mapping = {'Walking':0, 
                         'Jogging':1,
                          'Sitting':2,
                          'Standing':3, 
                          'Upstairs':4,
                          'Downstairs':5}
        df["activity_id"] = df["activity_id"].map(label_mapping)
        df["activity_id"] = df["activity_id"].map(self.labelToId)
        df["sub_id"] = df["sub_id"].astype(int)

        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            temp = df[df["sub_id"]==key]
            train_vali = pd.concat([train_vali,temp])

        test = pd.DataFrame()
        for key in self.test_keys:
            temp = df[df["sub_id"]==key]
            test = pd.concat([test,temp])
        
        # the col position varies between different datasets

        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,0]
        train_vali = train_vali.iloc[:,1:]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,0]  
        test = test.iloc[:,1:]

            
        return train_vali, train_vali_label, test, test_label
