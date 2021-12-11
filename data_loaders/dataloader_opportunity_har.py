import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA
# TODO the cols ! name
# ========================================      Opportunity_HAR_DATA         =============================
class Opportunity_HAR_DATA(BASE_DATA):
    """
    OPPORTUNITY Dataset for Human Activity Recognition from Wearable, Object, and Ambient Sensors
	
    Brief Description of the Dataset:
    ---------------------------------
    Each .dat file contains a matrix of data in text format. 
    Each line contains the sensor data sampled at a given time (sample rate: 30Hz). 
    For more detail . please reffer to the docomentation.html
    """
    def __init__(self, args):
        super(Opportunity_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)
        # the sensors between 134 and 248 are amounted on devices, so they will not to be considered
        # Specifically, the following columns were used for the challenge: 
        # =============================================================
        # 1-37, 38-46, 51-59, 64-72, 77-85, 90-98, 103-134, 244, 250.
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

        col_names         = ["dim_{}".format(i) for i in range(len(self.used_cols)-1)]
        self.col_names    =  col_names + ["activity_id"]

        self.train_keys   = [               'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat',  'S1-ADL5.dat', 'S1-Drill.dat', # subject 1
                             'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                                'S2-Drill.dat', # subject 2
                             'S3-ADL1.dat', 'S3-ADL2.dat',                                                               # subject 3
                             'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat',                 'S4-ADL5.dat', 'S4-Drill.dat'] # subject 4
        self.vali_keys    = [ 'S1-ADL1.dat','S3-ADL3.dat', 'S3-Drill.dat','S4-ADL4.dat',]
        self.test_keys    = ['S2-ADL4.dat', 'S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat']

        self.drop_activities = []
        
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
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()



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
