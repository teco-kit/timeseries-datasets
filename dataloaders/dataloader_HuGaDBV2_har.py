import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA
# TODO V1 and V2
# The second version of the dataset contains more precisely labeled labels. 
# The data has been re-labeled for the following activities: sitting, standing, sitting down, standing up.
# ========================================       HuGaDBV2_HAR_DATA             =============================
class HuGaDBV2_HAR_DATA(BASE_DATA):

    """

    https://github.com/romanchereshnev/HuGaDB
    HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks
    https://arxiv.org/abs/1705.08506

    BASIC INFO ABOUT THE DATA:
    ---------------------------------

    Sampling rate:  58 Hz
	
    The main data body of every file has 39 columns. 
    Each column corresponds to a sensor, and one row corresponds to a sample. 
    The order of the columns is fixed. 

    The first 36 columns correspond to the inertial sensors, 
    the next 2 columns correspond to the EMG sensors, 
    and the last column contains the activity ID. 

    Values of the gyroscopes and the accelerometers encoded by int_16 datatype. 
    Values of the EMGs encoded by uint_8 datatype.

    The inertial sensors are listed in the following order: 

        -- right foot (RF), 
        -- right shin (RS), 
        -- right thigh (RT), 
        -- left foot (LT), 
        -- left shin (LS), 
        -- and left thigh (LT), 
        -- followed by right EMG (R) 
        -- and left EMG (L). 

    Each inertial sensor produces three acceleration data on x,y,z axes and three gyroscope data on x,y,z axes.

    For instance, the column named 'RT_acc_z' contains data obtained from the z-axis of accelerometer located on the right thigh.

    Every file name was created according to the following template HGD_vX_ACT_PR_CNT.txt. 


    ID     Activity     Time (min)     Percent   Description
    1     walking         192          32.15       Walking and turning at various speed on flat surface.
    2     running         20            3.39       Running at various pace.
    3     going_up        37            6.23       Taking stairs up at various speed.
    4     going_down      33            5.52       Taking the stairs down at various speed and steps.
    5     sitting         68            11.45      Sitting on a chair; sitting on floor not included.
    6     sitting down    6             1.14       Sitting on a chair; sitting down on floor not included.
    7     standing up     6             1.06       Standing up from chair.
    8     standing        93            15.56      Static standing on solid surface.
    9     bicycling       44             7.41      Regular bicycling.
    10    up_by_elevator  25             4.22      Standing in elevator while moving up.
    11    down_by_elevator 19            3.30      Standing in elevator while moving down.
    12    sitting in car   51            8.55      Sitting while traveling by car. 
          Total           598           100.00 

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


        # The first 36 columns correspond to the inertial sensors, 
        # the next 2 columns correspond to the EMG sensors, 
        # and the last column contains the activity ID. 
        self.used_cols = [ 0,  1,  2,  3,  4,  5,   # right foot (RF)
                           6,  7,  8,  9, 10, 11,   # right shin (RS)
                           12, 13, 14, 15, 16, 17,  # right thigh (RT)
                           18, 19, 20, 21, 22, 23,  # left foot (LT)
                           24, 25, 26, 27, 28, 29,  # left shin (LS)
                           30, 31, 32, 33, 34, 35,  # left thigh (LT)
                           36, 37,  #right EMG (R)  Left EMG (L)
                           38 ] # label

        self.col_names    =  ['acc_rf_x', 'acc_rf_y', 'acc_rf_z', 'gyro_rf_x', 'gyro_rf_y', 'gyro_rf_z', 
                              'acc_rs_x', 'acc_rs_y', 'acc_rs_z', 'gyro_rs_x', 'gyro_rs_y', 'gyro_rs_z', 
                              'acc_rt_x', 'acc_rt_y', 'acc_rt_z', 'gyro_rt_x', 'gyro_rt_y', 'gyro_rt_z', 
                              'acc_lf_x', 'acc_lf_y', 'acc_lf_z', 'gyro_lf_x', 'gyro_lf_y', 'gyro_lf_z', 
                              'acc_ls_x', 'acc_ls_y', 'acc_ls_z', 'gyro_ls_x', 'gyro_ls_y', 'gyro_ls_z',
                              'acc_lt_x', 'acc_lt_y', 'acc_lt_z', 'gyro_lt_x', 'gyro_lt_y', 'gyro_lt_z', 
                              'EMG_r', 'EMG_l', 
                              'activity_id']

        # In Dataset V1, there is no 12:sitting in the car 
        # In Dataset V2, there is no 9:bicycling no 12 either
        self.label_map = [(1, 'walking'), 
                          (2, "running"),
                          (3, "going_up"), 
                          (4, "going_down"), 
                          (5, "sitting"), 
                          (6, "sitting down"), 
                          (7, "standing up"),
                          (8, "standing"), 
                          #(9, "bicycling"), 
                          (10, "up_by_elevator"),
                          (11, "down_by_elevator"),
                          #(12, "sitting in car")
                          ] 

        self.drop_activities = []


        # TODO This should be referenced by other paper
        # TODO , here the keys for each set will be updated in the readtheload function
        # All subjects IDs are : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        self.train_keys   = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18]
        self.vali_keys    = []
        self.test_keys    = [5,10,15]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(HuGaDBV2_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")

        root_path = os.path.join(root_path, "HuGaDB v2")

        file_list = os.listdir(root_path)
        # filtering out readme file
        file_list = [file for file in file_list if "HuGaDB" in file]

        df_dict = {}

        for file in file_list:

            sub_data = pd.read_csv(os.path.join(root_path,file), sep='\t',skiprows=(0,1,2))

            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # No missing Value

            sub = int(file.split("_")[-2])
            trial = file.split("_")[-1][:2]

            sub_id = "{}_{}".format(sub,trial)
            sub_data["sub_id"] = sub_id
            sub_data["sub"] = sub

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)  

            df_dict[sub_id] = sub_data

        # all data
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y