import pandas as pd
import numpy as np
import os
import random
import pywt
import pickle
from random import sample
from dataloaders.utils import Normalizer
from sklearn.utils import class_weight
# ========================================       Data loader Base class               =============================
class BASE_DATA():

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
        self.model_type   = args.model_type

        self.data_name    = args.data_name
        self.difference   = args.difference
        self.datanorm_type= args.datanorm_type
        self.train_vali_quote   = args.train_vali_quote

        # the following parameters are all for slidingwindow set up 
        self.freq         = args.sampling_freq  
        self.windowsize   = args.windowsize
        self.displacement = args.displacement


        self.wavename =  args.wavename
        self.freq_save_path = args.freq_save_path
        # Data Set spezifisch parameters
        # self.used_cols = []
		# self.train_keys   = []
        # self.vali_keys    =[]
        # self.test_keys    = []
        # self.drop_activities = []
        # self.col_names    =  []
        # self.file_encoding = {}
        # self.label_map = []
        print("About to load participants")
        self.participants = self.load_participants(self.root_path)

        
    def load_all_the_data(self, root_path):
        raise NotImplementedError

    def load_participants(self, root_path):
        print("Wrong load_participants")
        raise NotImplementedError


    def differencing(self, df):
        # columns = [, "acc_x"..."acc_y", "sub"], index is  sub_id
        # define the name for differenced columns
        sensor_cols = df.columns[:-1]
        columns = ["diff_"+i for i in sensor_cols]

        # The original data has been divided into segments by sub_id: a segment belongs to a same user 
        # There is no continuity between different segments, so diffrecne is only done within each segment

        # Train_vali_diff
        diff_data = []
        for id in df.index.unique():
            diff_data.append(df.loc[id,sensor_cols].diff())

        diff_data = pd.concat(diff_data)
        diff_data.columns = columns
        diff_data.fillna(method ="backfill",inplace=True)
        data = pd.concat([df.iloc[:,:-1],diff_data, df.iloc[:,-1]], axis=1)

        return data.reset_index()

    def normalization(self, train_vali, test=None):
        train_vali_sensors = train_vali.iloc[:,1:-1]
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors)
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors)
        train_vali_sensors = pd.concat([train_vali.iloc[:,0],train_vali_sensors,train_vali.iloc[:,-1]], axis=1)
        if test is None:
            return train_vali_sensors
        else:
            test_sensors  = test.iloc[:,1:-1]
            test_sensors  = self.normalizer.normalize(test_sensors)
            test_sensors  =  pd.concat([test.iloc[:,0],test_sensors,test.iloc[:,-1]], axis=1)
            return train_vali_sensors, test_sensors

    def get_the_sliding_index(self, data_x, data_y):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """
        print("----------------------- Get the Sliding Window -------------------")
        #data_x = data_x.reset_index()
        data_y = data_y.reset_index()

        data_x["activity_id"] = data_y["activity_id"]
        data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()


        freq         = self.freq   
        windowsize   = self.windowsize

        displacement = self.displacement


        window_index = []
        for index in data_x.act_block.unique():

            temp_df = data_x[data_x["act_block"]==index]
            assert len(temp_df["activity_id"].unique())==1

            if temp_df["activity_id"].unique()[0] not in self.drop_activities:
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]
                end   = start+windowsize

                while end <= temp_df.index[-1] + 1 :

                    window_index.append([sub_id, start, end])

                    start = start + displacement
                    end   = start + windowsize


        return window_index

    def genarate_spectrogram(self):
        save_path = os.path.join(self.freq_save_path,self.data_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.freq_path = os.path.join(save_path,"diff_{}_window_{}_step_{}".format(self.difference, self.windowsize,self.displacement))
        if os.path.exists(self.freq_path):
            print("----------------------- file are generated -----------------------")
            with open(os.path.join(self.freq_path,"freq_file_name.pickle"), 'rb') as handle:
                self.freq_file_name = pickle.load(handle)
        else:
            print("----------------------- spetrogram generating -----------------------")
            os.mkdir(self.freq_path)
            totalscal = self.freq + 1
            fc = pywt.central_frequency(self.wavename)#计算小波函数的中心频率
            cparam = 2 * fc * totalscal  #常数c
            scales = cparam/np.arange(totalscal,1,-1) #为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
            self.freq_file_name = []

            temp_data = self.normalization(self.data_x.copy())
            for window in self.slidingwindows:
                sub_id = window[0]
                start_index = window[1]
                end_index = window[2]
	
                name = "{}_{}_{}".format(sub_id,start_index,end_index)
                self.freq_file_name.append(name)

                sample_x = temp_data.iloc[start_index:end_index,1:-1].values
                scalogram = []
                for j in range(sample_x.shape[1]):
                    [cwtmatr, frequencies] = pywt.cwt(sample_x[:,j],   scales,   self.wavename,sampling_period = 1.0/self.freq)#连续小波变换模块
                    scalogram.append(cwtmatr)
                scalogram = np.stack(scalogram)
                with open(os.path.join(self.freq_path,"{}.pickle".format(name)), 'wb') as handle:
                    pickle.dump(scalogram, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.freq_path,"freq_file_name.pickle"), 'wb') as handle:
                pickle.dump(self.freq_file_name, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_classes_weight(self):
        class_transform = {x: i for i, x in enumerate(self.no_drop_activites)}

        y_of_all_windows  = []
        for window in self.slidingwindows:
            start_index = window[1]
            end_index = window[2]
            y_of_all_windows.append(class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]])
        act_weights = class_weight.compute_class_weight(class_weight='balanced',classes=range(len(self.no_drop_activites)),y=y_of_all_windows)
        act_weights = act_weights.round(4)
        return act_weights