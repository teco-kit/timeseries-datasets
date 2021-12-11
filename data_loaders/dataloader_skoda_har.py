import pandas as pd
import numpy as np
import os
import scipy.io as sio

from data_loaders.dataloader_base import BASE_DATA
# TODO Train TEST split
# 左右手！
# ========================================       Skoda_HAR_DATA               =============================
class Skoda_HAR_DATA(BASE_DATA):

    """
    Activity recognition dataset - Skoda Mini Checkpoint
    Brief Description of the Dataset:
    ---------------------------------

    Sensors
    This dataset contains 10 classes, recorded with a 2x10 USB sensors placed on the left and right upper and lower arm.

    Sensor sample rate is approximately 98Hz.
    The locations of the sensors on the arms is documented in the figure.

    right_classall_clean.mat and left_classall_clean.mat: matlab .mat files with original datafor right and left arm sensors

    label value:
        32 null class
        48 write on notepad
        49 open hood
        50 close hood
        51 check gaps on the front door
        52 open left front door
        53 close left front door
        54 close both left door
        55 check trunk gaps
        56 open and close trunk
        57 check steering wheel
    """

    def __init__(self, args):
        super(Skoda_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

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
                train_vali_act_block = act_block_list[:int(0.8*len(act_block_list))]  # test_act_blocks = act_block_list[int(0.8*len(act_block_list)):]

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