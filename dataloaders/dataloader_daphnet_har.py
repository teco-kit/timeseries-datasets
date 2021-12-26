import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================    Daphnet_HAR_DATA        =============================
class Daphnet_HAR_DATA(BASE_DATA):

    """
    BASIC INFO ABOUT THE DATA:
    ---------------------------------
    The dataset comprises 3 wearable wireless acceleration sensors (see [10] for sensor details) recording 3D acceleration at 64 Hz. 
    The sensors are placed at the ankle (shank), on the thigh just above the knee, and on the hip.

    0: not part of the experiment. For instance the sensors are installed on the user or the user is performing activities unrelated to the experimental protocol, such as debriefing
    1: experiment, no freeze (can be any of stand, walk, turn)
    2: freeze
    """

    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)

        # Time of sample in millisecond
        # Ankle (shank) acceleration - horizontal forward acceleration [mg]
        # Ankle (shank) acceleration - vertical [mg]
        # Ankle (shank) acceleration - horizontal lateral [mg]
        # Upper leg (thigh) acceleration - horizontal forward acceleration [mg]
        # Upper leg (thigh) acceleration - vertical [mg]
        # Upper leg (thigh) acceleration - horizontal lateral [mg]
        # Trunk acceleration - horizontal forward acceleration [mg]
        # Trunk acceleration - vertical [mg]
        # Trunk acceleration - horizontal lateral [mg]
        # Annotations (see Annotations section)

        # the first cols no use
        self.used_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.col_names    =  ["acc_h_f_ankle", "acc_v_ankle", "acc_h_l_ankle",
                              "acc_h_f_leg", "acc_v_leg", "acc_h_l_leg",
                              "acc_h_f_trunk", "acc_v_trunk","acc_h_l,trunk",
                              "activity_id"]

        self.label_map = [
            (0, 'Other'),
            (1, 'No freeze'),
            (2, 'Freeze')
        ]

        self.drop_activities = [0]

        self.train_keys   = [11, 12, # 'S01R01.txt', 'S01R02.txt',
                             22, #'S02R02.txt', 
                             31, 32, # 'S03R01.txt', 'S03R02.txt',
                             33, # 'S03R03.txt', 
                             41, # 'S04R01.txt',
                             51, #'S05R01.txt' ]
                             52,  #'S05R02.txt'
                             61, 62, # 'S06R01.txt', 'S06R02.txt',
                             71, 72, # 'S07R01.txt', 'S07R02.txt',
                             81, #'S08R01.txt', 
                             91, # 'S09R01.txt', 
                             101 ] #'S10R01.txt' 

        self.vali_keys    =[]

        self.test_keys    = [21, # 'S02R01.txt',
                             22] #'S02R02.txt', 


        self.exp_mode     = args.exp_mode
        if self.exp_mode == "LOCV":
            self.split_tag = "sub"
        else:
            self.split_tag = "sub_id"

        self.LOCV_keys = [[1,2],[3,4],[4,6],[7,8],[9,10]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10]
        self.sub_ids_of_each_sub = {}


        self.file_encoding = {'S01R01.txt':11, 'S01R02.txt':12,
                              'S02R01.txt':21, 'S02R02.txt':22,
                              'S03R01.txt':31, 'S03R02.txt':32, 'S03R03.txt':33,
                              'S04R01.txt':41,
                              'S05R01.txt':51, 'S05R02.txt':52,
                              'S06R01.txt':61, 'S06R02.txt':62,
                              'S07R01.txt':71, 'S07R02.txt':72,
                              'S08R01.txt':81,
                              'S09R01.txt':91,
                               'S10R01.txt':101}
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(Daphnet_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        file_list = os.listdir(root_path)

        assert len(file_list) == 17
        df_dict = {}

        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file),header=None, delim_whitespace=True)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # TODO check missing labels? 
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            # This data set dose not need label transformation

            sub = int(str(self.file_encoding[file])[:-1])
            sub_data['sub_id'] = self.file_encoding[file]
            sub_data["sub"] = sub
            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(self.file_encoding[file])

            df_dict[self.file_encoding[file]] = sub_data


        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')
        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]


        # it is not necessary here
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
            
        return data_x, data_y