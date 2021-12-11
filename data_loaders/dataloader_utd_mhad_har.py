import pandas as pd
import numpy as np
import os
from random import sample
import scipy.io as io
from data_loaders.utils import Normalizer

# https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

# ========================================       UTD_MHAD_HAR_DATA               =============================
class UTD_MHAD_HAR_DATA():

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




        self.used_cols = [0,1,2,3,4,5]

        # there are total 6 sensors 

        self.col_names = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']


        # TODO 
        self.train_keys   = [1,2,3,4,5,6,]

        self.vali_keys    = [7]

        self.test_keys    = [8]

        self.drop_activities = [22,23,24,25,26,27]
        # or self.drop_activities = list(np.arange(1,22))

        # Our UTD-MHAD dataset consists of 27 different actions: (1) right arm swipe to the left, (2) right arm swipe to the right, 
        # (3) right hand wave, (4) two hand front clap, (5) right arm throw, (6) cross arms in the chest, (7) basketball shoot, 
        # (8) right hand draw x, (9) right hand draw circle (clockwise), (10) right hand draw circle (counter clockwise), 
        # (11) draw triangle, (12) bowling (right hand), (13) front boxing, (14) baseball swing from right, 
        # (15) tennis right hand forehand swing, (16) arm curl (two arms), (17) tennis serve, (18) two hand push, 
        # (19) right hand knock on door, (20) right hand catch an object, (21) right hand pick up and throw, (22) jogging in place, 
        # (23) walking in place, (24) sit to stand, (25) stand to sit, (26) forward lunge (left foot forward), (27) squat (two arms stretch out).

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


    def read_data(self):

        train_vali_x, train_vali_y, test_x, test_y = self.load_the_data(root_path     = self.root_path)

        if self.difference:
            train_vali_x, test_x = self.differencing(train_vali_x, test_x)
            
        if self.datanorm_type is not None:
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

        train_vali_window_index = self.get_the_sliding_index(train_vali_x.copy(), train_vali_y.copy(), Flag_id = True)
        self.test_window_index = self.get_the_sliding_index(test_x.copy(), test_y.copy(), Flag_id = False)

        self.train_window_index, self.vali_window_index =  self.train_vali_split(train_vali_window_index)

        self.train_vali_x = train_vali_x.copy()
        self.train_vali_y = train_vali_y.copy()

        self.test_x = test_x.copy()
        self.test_y = test_y.copy()





    def load_the_data(self, root_path):
        file_list = os.listdir(root_path)

        temp_train_keys = []
        temp_test_keys = []
        temp_vali_keys = []

        df_dict = {}
        a = []
        for file in file_list:
            file_split = file.split("_")
            action = int(file_split[0][1:])
            sub = int(file_split[1][1:])
            trial =  int(file_split[2][1:])
            sub_id = "{}_{}_{}".format(sub, action, trial)

            sub_data =io.loadmat(os.path.join(root_path,file))["d_iner"]
            sub_data = pd.DataFrame(sub_data,columns=self.col_names )  

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

        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        train_vali = df_all.loc[self.train_keys+self.vali_keys]
        test = df_all.loc[self.test_keys]

        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]
        test = test.iloc[:,:-1]

        return train_vali, train_vali_label, test, test_label


    def differencing(self, train_vali, test):
        # define the name for differenced columns
        columns = ["diff_"+i for i in train_vali.columns]
        # The original data has been divided into segments by the sliding window method. 
        # There is no continuity between paragraphs, so diffrecne is only done within each segment

        # Train_vali_diff

        diff_train_vali = []
        for id in train_vali.index.unique():
            diff_train_vali.append(train_vali.loc[id].diff())
        diff_train_vali = pd.concat(diff_train_vali)
        diff_train_vali.columns = columns
        diff_train_vali.fillna(method ="backfill",inplace=True)
        train_vali = pd.concat([train_vali,diff_train_vali], axis=1)


        diff_test = []
        for id in test.index.unique():
            diff_test.append(test.loc[id].diff())
        diff_test = pd.concat(diff_test)
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

    def get_the_sliding_index(self, data_x, data_y, Flag_id = True):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """
        #data_x.index = [1]*data_x.shape[0]
        #data_x.index.name = "sub_id"
        data_x = data_x.reset_index()
        data_y = data_y.reset_index()

        data_x["activity_id"] = data_y["activity_id"]
        data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()


        # TODO s!!!!!!!!!!!!!!!   Dataset Dependent!!!!!!!!!!!!!!!!!!!! 
        # To set the window size and Sliding step
        freq         = 50  
        windowsize   = int(1 * freq)

        if Flag_id:
            displacement = int(1 * freq)
        else:
            displacement = int(1 * freq)

        drop_long    = 3
        window_index = []
        drop_ubergang = False

        if drop_ubergang == True:
            drop_index = []
            numblocks = data_x['act_block'].max()
            for block in range(1, numblocks+1):
                drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
                drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
            data_x = data_x.drop(drop_index)

        for index in data_x.act_block.unique():
            temp_df = data_x[data_x["act_block"]==index]
            assert len(temp_df["activity_id"].unique())==1
            if temp_df["activity_id"].unique()[0] not in self.drop_activities:
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]
                end   = start+windowsize

                while end <= temp_df.index[-1]+1:
                    if Flag_id:
                        window_index.append([sub_id, start, end])
                    else:
                        window_index.append([start, end])

                    start = start + displacement
                    end   = end + displacement

        return window_index

    def train_vali_split(self, train_vali_window_index):
        """
        if vali_keys is not None ----> subject split 
        if vali_keys is None ------> random 80% split
        After train test split : the window index consists of only TWO components : start_index and end_index
        """
        train_window_index = []
        vali_window_index  = []
        if len(self.vali_keys):
            print("Subjects Split")
            if len(self.file_encoding)>0:
                vali_keys    = [self.file_encoding[key] for key in self.vali_keys]
            else:
                vali_keys = self.vali_keys

            for item in train_vali_window_index:
                sub_id = item[0]
                if sub_id in vali_keys:
                    vali_window_index.append([item[1],item[2]])
                else:
                    train_window_index.append([item[1],item[2]])
        else:
            print("Random split")
            all_index = list(np.arange(len(train_vali_window_index)))
            train_list = sample(all_index,int(len(all_index)*0.8))
            for i in all_index:
                item = train_vali_window_index[i]
                if i in train_list:
                    train_window_index.append([item[1],item[2]])
                else:
                    vali_window_index.append([item[1],item[2]])

        return train_window_index, vali_window_index

