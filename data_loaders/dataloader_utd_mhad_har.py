import pandas as pd
import numpy as np
import os

import scipy.io as io
from data_loaders.dataloader_base import BASE_DATA
# ========================================       UTD_MHAD_HAR_DATA               =============================
class UTD_MHAD_HAR_DATA(BASE_DATA):
    """
    https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
    UTD Multimodal Human Action Dataset (UTD-MHAD)

    BASIC INFO ABOUT THE DATA:
    ---------------------------------

    The dataset contains 27 actions performed by 8 subjects (4 females and 4 males).

    Our UTD-MHAD dataset consists of 27 different actions: 
        (1) right arm swipe to the left, 
        (2) right arm swipe to the right, 
        (3) right hand wave, (4) two hand front clap, 
        (5) right arm throw, (6) cross arms in the chest, 
        (7) basketball shoot, (8) right hand draw x, 
        (9) right hand draw circle (clockwise), 
        (10) right hand draw circle (counter clockwise), 
        (11) draw triangle, (12) bowling (right hand), 
        (13) front boxing, (14) baseball swing from right, 
        (15) tennis right hand forehand swing, (16) arm curl (two arms), 
        (17) tennis serve, (18) two hand push, (19) right hand knock on door, 
        (20) right hand catch an object, (21) right hand pick up and throw, 
        (22) jogging in place, (23) walking in place, (24) sit to stand, 
        (25) stand to sit, (26) forward lunge (left foot forward), 
        (27) squat (two arms stretch out).
    The inertial sensor was worn on the subject's right wrist or the right thigh (see the figure below) 
    depending on whether the action was mostly an arm or a leg type of action. 
    Specifically, for actions 1 through 21, the inertial sensor was placed on the subject's right wrist; 
    for actions 22 through 27, the inertial sensor was placed on the subject's right thigh.

    The sampling rate of this wearable inertial sensor is 50 Hz. 
    The measuring range of the wearable inertial sensor is ±8g for acceleration and ±1000 degrees/second for rotation.
    """
    def __init__(self, args):
        super(UTD_MHAD_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """


        self.used_cols = [0,1,2,3,4,5]
        # there are total 6 sensors 
        self.col_names = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']


        # TODO 
        # TODO , here the keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3,4,5,6,]
        self.vali_keys    = [7]
        self.test_keys    = [8]

        # TODO 2 sub sets!
        self.drop_activities = [22,23,24,25,26,27]
        # or self.drop_activities = list(np.arange(1,22))

        self.file_encoding = {}  # no use 
        
        self.label_map = [(1, 'right arm swipe to the left'), 
                          (2, "right arm swipe to the right"),
                          (3, "right hand wave"), 
                          (4, "two hand front clap"), 
                          (5, "right arm throw"), 
                          (6, "cross arms in the chest"), 
                          (7, "basketball shoot"),
                          (8, "right hand draw x"), 
                          (9, "right hand draw circle (clockwise)"), 
                          (10, "right hand draw circle (counter clockwise)"),
                          (11, "draw triangle"),
                          (12, "bowling (right hand)"), 
                          (13, "front boxing"), 
                          (14, "baseball swing from right"),
                          (15, "tennis right hand forehand swing"), 
                          (16, "arm curl (two arms)"), 
                          (17, "tennis serve"), 
                          (18, "two hand push"), 
                          (19, "right hand knock on door"),
                          (20, "right hand catch an object"), 
                          (21, "right hand pick up and throw"), 
                          (22, "jogging in place"),
                          (23, "walking in place"), 
                          (24, "sit to stand"), 
                          (25, "stand to sit"), 
                          (26, "forward lunge (left foot forward)"), 
                          (27, "squat (two arms stretch out)")] 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

    def load_the_data(self, root_path):
        file_list = os.listdir(root_path)

        temp_train_keys = []
        temp_test_keys = []
        temp_vali_keys = []

        df_dict = {}

        for file in file_list:
            file_split = file.split("_")

            action = int(file_split[0][1:])
            sub = int(file_split[1][1:])
            trial =  int(file_split[2][1:])

            sub_data =io.loadmat(os.path.join(root_path,file))["d_iner"]
            sub_data = pd.DataFrame(sub_data)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            sub_id = "{}_{}_{}".format(sub, action, trial)
            sub_data["sub_id"] = sub_id

            sub_data["activity_id"] = action 

            if sub in self.train_keys:
                temp_train_keys.append(sub_id)
            elif sub in self.vali_keys:
                temp_vali_keys.append(sub_id)        
            else:
                temp_test_keys.append(sub_id)      

            df_dict[sub_id] = sub_data

        self.train_keys  = temp_train_keys 
        self.test_keys  = temp_test_keys 
        self.vali_keys  = temp_vali_keys 


        df_all = pd.concat(df_dict)
        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # train_vali Test split 
        train_vali = df_all.loc[self.train_keys+self.vali_keys]
        test = df_all.loc[self.test_keys]

        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label
