import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA
# ========================================       FUSION_HAR_DATA             =============================
class FUSION_HAR_DATA(BASE_DATA):
    """
    Fusion of Smartphone Motion Sensors for Physical Activity Recognition
    https://www.utwente.nl/en/eemcs/ps/research/dataset/

    Brief Description of the Dataset:
    ---------------------------------
    In the data collection experiments, we collected data for seven physical activities. 
    These are walking, sitting, standing, jogging, biking, walking upstairs and walking downstairs, 
    which are mainly used in the related studies and they are the basic motion activities in daily life.
    There were ten participants involved in our data collection experiment who performed each of these activities for 3-4 minutes. 
    All ten participants were male, between the ages of 25 and 30. 
    The experiments were carried out indoors in one of the university buildings, except biking. 
    For walking, and jogging, the department corridor was used. 
    For walking upstairs and downstairs, a 5-floor building with stairs was used. 
    Each of these participants was equipped with five smartphones on five body positions: 
        One in their right jean pocket. 
        One in their left jean pocket.
        One on belt position towards the right leg using a belt clipper.
        One on the right upper arm. 
        One on the right wrist. 
    The data was recorded for all five positions at the same time for each activity and it was collected at a rate of 50 samples per second. 
    This sampling rate (50 samples per second) is enough to recognize human physical activities, as we show in our previous study . 
    Moreover, in the state of the art, frequencies lower than 50 samples per second have been shown to be sufficient for activity recognition.
    """

    def __init__(self, args):
        super(FUSION_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """



        # Because there are empty cols 
        self.used_cols = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, # Left_pocket
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, # Right_pocket
                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, # Wrist
                         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # Upper_arm
                         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Belt
                         69]

        col_list    =  ["acc_x","acc_y","acc_z","lacc_x","lacc_y","lacc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
        pos_list = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
        self.col_names    =  [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]+["activity_id"]


        # TODO This should be referenced by other paper
        self.train_keys   = ["Participant_1.csv", "Participant_2.csv", "Participant_3.csv",
                             "Participant_4.csv", "Participant_5.csv", "Participant_6.csv"]
        self.vali_keys    = ["Participant_7.csv", "Participant_8.csv"]
        self.test_keys    = ["Participant_9.csv", "Participant_10.csv"]

        self.drop_activities = []

        # The original file used the str as the label, they need first encoded
        self.label_map = [(0, 'walking'), 
                          (1, 'standing'),
                          (2, 'jogging'),
                          (3, 'sitting'), 
                          (4, 'biking'),
                          (5, 'upstairs'),
                          (6, "downstairs")]

        self.file_encoding = {"Participant_1.csv":1, "Participant_2.csv":2, "Participant_3.csv":3,
                              "Participant_4.csv":4, "Participant_5.csv":5, "Participant_6.csv":6,
                              "Participant_7.csv":7, "Participant_8.csv":8, "Participant_9.csv":9,
                              "Participant_10.csv":10} 


        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

    def load_the_data(self, root_path):

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if "Participant" in file] # in total , it should be 10

        assert len(file_list) == 10

        df_dict = {}

        for file in file_list:
            sub_data = pd.read_csv(os.path.join(root_path, file), skiprows=[0,1], header=None)
            sub_data =sub_data.iloc[:,self.used_cols]

            sub_data.columns = self.col_names

            # if missing values, imputation TODO
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')

            sub_data['sub_id'] =int(self.file_encoding[file])

            df_dict[self.file_encoding[file]] = sub_data
        # this is for first encode the str label to nummeric number
        label_mapping = {'walking':0, 
                         'standing':1,
                         'jogging':2,
                         'sitting':3, 
                         'biking':4,
                         'upstairs':5,
                         "downstairs":6}
        # there are typos in the data!!!
        def change_the_typo(label):
            if label == "upsatirs":
                return "upstairs"
            else:
                return label

        # Train Vali
        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            train_vali = pd.concat([train_vali,df_dict[self.file_encoding[key]]])

        train_vali["activity_id"] = train_vali["activity_id"].apply(change_the_typo)
        train_vali["activity_id"] = train_vali["activity_id"].map(label_mapping)
        train_vali["activity_id"] = train_vali["activity_id"].map(self.labelToId)

        # Test
        test = pd.DataFrame()
        for key in self.test_keys:
            test = pd.concat([test,df_dict[self.file_encoding[key]]])

        test["activity_id"] = test["activity_id"].apply(change_the_typo)
        test["activity_id"] = test["activity_id"].map(label_mapping)
        test["activity_id"] = test["activity_id"].map(self.labelToId)


        # X y
        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]  
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label