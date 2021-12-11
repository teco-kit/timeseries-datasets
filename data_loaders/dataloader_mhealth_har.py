import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA
# ========================================   Mhealth_HAR_DATA      =============================
class Mhealth_HAR_DATA(BASE_DATA):

    """

    Brief Description of the Dataset:
    ---------------------------------
    1) Experimental Setup

    The collected dataset comprises body motion and vital signs recordings for ten volunteers of diverse profile while performing 12 physical activities (Table 1). 
    Shimmer2 [BUR10] wearable sensors were used for the recordings. The sensors were respectively placed on the subject's chest, right wrist and left ankle and 
    attached by using elastic straps (as shown in the figure in attachment). The use of multiple sensors permits us to measure the motion experienced by diverse body parts, 
    namely, the acceleration, the rate of turn and the magnetic field orientation, thus better capturing the body dynamics. The sensor positioned
    on the chest also provides 2-lead ECG measurements which are not used for the development of the recognition model but rather collected for future work purposes.
    This information can be used, for example, for basic heart monitoring, checking for various arrhythmias or looking at the effects of exercise on the ECG. 

    All sensing modalities are recorded at a sampling rate of 50 Hz, which is considered sufficient for capturing human activity. Each session was recorded using a video camera.

    This dataset is found to generalize to common activities of the daily living, given the diversity of body parts involved in each one (e.g., frontal elevation of arms vs.
    knees bending), the intensity of the actions (e.g., cycling vs. sitting and relaxing) and their execution speed or dynamicity (e.g., running vs. standing still). The activities
    were collected in an out-of-lab environment with no constraints on the way these must be executed, with the exception that the subject should try their best when executing them.

    2) Activity set

    The activity set is listed in the following:

    L1: Standing still (1 min) 
    L2: Sitting and relaxing (1 min) 
    L3: Lying down (1 min) 
    L4: Walking (1 min) 
    L5: Climbing stairs (1 min) 
    L6: Waist bends forward (20x) 
    L7: Frontal elevation of arms (20x)
    L8: Knees bending (crouching) (20x)
    L9: Cycling (1 min)
    L10: Jogging (1 min)
    L11: Running (1 min)
    L12: Jump front & back (20x)
    """

    def __init__(self, args, flag="train"):
        super(Mhealth_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # Column 1: acceleration from the chest sensor (X axis)
        # Column 2: acceleration from the chest sensor (Y axis)
        # Column 3: acceleration from the chest sensor (Z axis)
        # Column 4: electrocardiogram signal (lead 1) 
        # Column 5: electrocardiogram signal (lead 2)
        # Column 6: acceleration from the left-ankle sensor (X axis)
        # Column 7: acceleration from the left-ankle sensor (Y axis)
        # Column 8: acceleration from the left-ankle sensor (Z axis)
        # Column 9: gyro from the left-ankle sensor (X axis)
        # Column 10: gyro from the left-ankle sensor (Y axis)
        # Column 11: gyro from the left-ankle sensor (Z axis)
        # Column 13: magnetometer from the left-ankle sensor (X axis)
        # Column 13: magnetometer from the left-ankle sensor (Y axis)
        # Column 14: magnetometer from the left-ankle sensor (Z axis)
        # Column 15: acceleration from the right-lower-arm sensor (X axis)
        # Column 16: acceleration from the right-lower-arm sensor (Y axis)
        # Column 17: acceleration from the right-lower-arm sensor (Z axis)
        # Column 18: gyro from the right-lower-arm sensor (X axis)
        # Column 19: gyro from the right-lower-arm sensor (Y axis)
        # Column 20: gyro from the right-lower-arm sensor (Z axis)
        # Column 21: magnetometer from the right-lower-arm sensor (X axis)
        # Column 22: magnetometer from the right-lower-arm sensor (Y axis)
        # Column 23: magnetometer from the right-lower-arm sensor (Z axis)
        # Column 24: Label (0 for the null class)

        self.used_cols = list(np.arange(24))
        self.col_names    =  ["acc_chest_x", "acc_chest_y" , "acc_chest_z",
                              "ecg_lead_1" , "ecg_lead_2",
                              "acc_left_ankle_x", "acc_left_ankle_y", "acc_left_ankle_z",
                              "gyro_left_ankle_x", "gyro_left_ankle_y", "gyro_left_ankle_z",
                              "mag_left_ankle_x", "mag_left_ankle_y", "mag_left_ankle_z",
                              "acc_right_lower_arm_x", "acc_right_lower_arm_y", "acc_right_lower_arm_z",
                              "gyro_right_lower_arm_x", "gyro_right_lower_arm_y", "gyro_right_lower_arm_z",
                              "mag_right_lower_arm_x", "right_lower_arm_y", "right_lower_arm_z",
                              "activity_id"]

        # TODO find the paper
        self.train_keys   = ['mHealth_subject1.log',
                             'mHealth_subject2.log',
                             'mHealth_subject3.log',
                             'mHealth_subject4.log',
                             'mHealth_subject5.log',
                             'mHealth_subject6.log']
        self.vali_keys    = ['mHealth_subject7.log',
                             'mHealth_subject8.log']
        self.test_keys    = ['mHealth_subject9.log',
                             'mHealth_subject10.log']

        self.drop_activities = [0]

        self.label_map = [(0, "other"),
                          (1, "Standing still" ),
                          (2, "Sitting and relaxing" ),
                          (3, "Lying down" ),
                          (4, "Walking" ),
                          (5, "Climbing stairs"),
                          (6, "Waist bends forward" ),
                          (7, "Frontal elevation of arms" ),
                          (8, "Knees bending (crouching)" ),
                          (9, "Cycling" ),
                          (10, "Jogging" ),
                          (11, "Running" ),
                          (12, "Jump front & back" )]

        self.file_encoding = {'mHealth_subject1.log':1,
                              'mHealth_subject2.log':2,
                              'mHealth_subject3.log':3,
                              'mHealth_subject4.log':4,
                              'mHealth_subject5.log':5,
                              'mHealth_subject6.log':6,
                              'mHealth_subject7.log':7,
                              'mHealth_subject8.log':8,
                              'mHealth_subject9.log':9,
                              'mHealth_subject10.log':10}


        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()


    def load_the_data(self, root_path):

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if "subject" in file] # in total , it should be 10
        
        assert len(file_list) == 10

        df_dict = {}

        for file in file_list:
            sub_data = pd.read_csv(os.path.join(root_path,file), sep = '\\\t', engine= 'python', header = None)

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