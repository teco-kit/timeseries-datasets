import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA
# ============================== UCI_HAR_DATA ======================================
class UCI_HAR_DATA(BASE_DATA):
    """
    The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. 
    Each person performed six activities 
    (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) 
    wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, 
    we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. 
    The experiments have been video-recorded to label the data manually. 
    The obtained dataset has been randomly partitioned into two sets, 
    where 70% of the volunteers was selected for generating the training data and 30% the test data. 

    The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters 
    and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). 
    The sensor acceleration signal, which has gravitational and body motion components, 
    was separated using a Butterworth low-pass filter into body acceleration and gravity. 
    The gravitational force is assumed to have only low frequency components, 
    therefore a filter with 0.3 Hz cutoff frequency was used. 
    From each window, a vector of features was obtained by calculating variables from the time and frequency domain. 
    See 'features_info.txt' for more details. 

        1 WALKING
        2 WALKING_UPSTAIRS
        3 WALKING_DOWNSTAIRS
        4 SITTING
        5 STANDING
        6 LAYING
    """
    def __init__(self, args):
        super(UCI_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """
        self.used_cols    = [] #no use , because this data format has save each sensor in one file
        self.col_names   =  ['body_acc_x_', 'body_acc_y_', 'body_acc_z_',
                             'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
                             'total_acc_x_', 'total_acc_y_', 'total_acc_z_']

        self.label_map = [ 
            (1, 'WALKING'),
            (2, 'WALKING_UPSTAIRS'),
            (3, 'WALKING_DOWNSTAIRS'),
            (4, 'SITTING'),
            (5, 'STANDING'),
            (6, 'LAYING'),
        ]

        self.drop_activities = []

        # All ID used for training [ 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
        # All ID used for Test  [ 2,  4,  9, 10, 12, 13, 18, 20, 24]
        self.train_keys   = [ 1,  3,  5,  7,  8, 14, 15, 16, 17, 21, 22, 23, 26, 27, 28, 29]
        self.vali_keys    = [ 6, 11, 19, 25, 30]
        self.test_keys    = [ 2,  4,  9, 10, 12, 13, 18, 20, 24]
		
        self.file_encoding = {}


        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

        #self.spectrogram  = args.spectrogram
        #if self.spectrogram:
        #    self.scales       = np.arange(1, int(args.f_max/2)+1)
        #    self.wavelet      = args.wavelet



    def load_the_data(self, root_path):

        temp_train_keys = []
        temp_test_keys = []
        temp_vali_keys = []
        # ====================  Load the sensor values ====================
        train_vali_path = os.path.join(root_path, "train/Inertial Signals/")
        test_path  = os.path.join(root_path, "test/Inertial Signals/")

        train_vali_dict = {}
        test_dict  = {}

        file_list = os.listdir(train_vali_path)
        for file in file_list:

            train_vali = pd.read_csv(train_vali_path + file,header=None, delim_whitespace=True)
            test  = pd.read_csv(test_path+file[:-9]+"test.txt",header=None, delim_whitespace=True)
			
            train_vali_dict[file[:-9]] = train_vali
            test_dict[file[:-9]] = test


        # =================== Define the sub id  and the label for each segments FOR  TRAIN VALI  ================
        train_vali = pd.DataFrame(np.stack([train_vali_dict[col].values.reshape(-1) for col in self.col_names], axis=1), columns = self.col_names)

        train_vali_subjects = pd.read_csv(os.path.join(root_path,"train/subject_train.txt"), header=None)
        train_vali_subjects.columns = ["subjects"]

        train_vali_label = pd.read_csv(os.path.join(root_path,"train/y_train.txt"),header=None)
        train_vali_label.columns = ["labels"]

        index = []
        labels = []

        assert train_vali_dict["body_acc_x_"].shape[0] == train_vali_subjects.shape[0]

        # repeat the id and the label for each segs 128 tims
        for i in range(train_vali_dict["body_acc_x_"].shape[0]):
            sub = train_vali_subjects.loc[i,"subjects"]
            sub_id = "{}_{}".format(sub,i)

            ac_id = train_vali_label.loc[i,"labels"]
            # according to the sub, add the sub id in to the train vali test keys
            if sub in self.train_keys:
                temp_train_keys.append(sub_id)
            elif sub in self.vali_keys:
                temp_vali_keys.append(sub_id)
            else:
                temp_test_keys.append(sub_id)

            index.extend(128*[sub_id])
            labels.extend(128*[ac_id])

        train_vali["sub_id"] = index
        train_vali["activity_id"] = labels

        # =================== Define the sub id  and the label for each segments  FOR TEST ================
        test = pd.DataFrame(np.stack([test_dict[col].values.reshape(-1) for col in self.col_names], axis=1), columns = self.col_names)

        test_subjects = pd.read_csv(os.path.join(root_path,"test/subject_test.txt"), header=None)
        test_subjects.columns = ["subjects"]

        test_label = pd.read_csv(os.path.join(root_path,"test/y_test.txt"),header=None)
        test_label.columns = ["labels"]

        index = []
        labels = []

        assert test_dict["body_acc_x_"].shape[0] == test_subjects.shape[0]

        for i in range(test_dict["body_acc_x_"].shape[0]):
            sub = test_subjects.loc[i,"subjects"]
            sub_id = "{}_{}".format(sub,i)

            ac_id = test_label.loc[i,"labels"]

            if sub in self.train_keys:
                temp_train_keys.append(sub_id)
            elif sub in self.vali_keys:
                temp_vali_keys.append(sub_id)
            else:
                temp_test_keys.append(sub_id)

            index.extend(128*[sub_id])
            labels.extend(128*[ac_id])

        test["sub_id"] = index
        test["activity_id"] = labels

        # ================= Updata the keys =================
        self.train_keys  = temp_train_keys 
        self.test_keys  = temp_test_keys 
        self.vali_keys  = temp_vali_keys 

        # The split may be different as the default setting, so we concat all segs together
        df_all = pd.concat([train_vali,test])
        df_dict = {}
        for i in df_all.groupby("sub_id"):
            df_dict[i[0]] = i[1]
        df_all = pd.concat(df_dict)

        # ================= Label Transformation ===================
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