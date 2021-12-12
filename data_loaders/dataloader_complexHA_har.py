import pandas as pd
import numpy as np
import os

from data_loaders.dataloader_base import BASE_DATA

# ========================================    ComplexHA_HAR_DATA        =============================
class ComplexHA_HAR_DATA(BASE_DATA):

    """
    Complex human activity recognition using smartphone and wrist-worn motion sensors
    This dataset contains different smartphone sensors data for 13 human activities 
    (walking, jogging, sitting, standing, biking, using stairs, typing, drinking coffee, eating, giving a talk, and smoking). 
    The details about the dataset and its collection process is described in the readme file
    BASIC INFO ABOUT THE DATA:
    ---------------------------------
    Labels for the excel sheet columns are in the following order: 

    Time stamp, accelerometer(x,y,z), linear acceleration sensor(x,y,z), gyroscope (x,y,z), Magnetometer (x,y,z), Activity Label(11111,11112, etc)

    Activity Label code: 

    walk        11111
    stand       11112
    jog         11113
    sit         11114
    bike        11115
    upstairs    11116
    downstairs  11117
    type        11118
    write       11119
    coffee      11120
    talk        11121
    smoke       11122
    eat         11123

    We collected a dataset for thirteen human activities. We selected these activities because they can
    be used for detecting bad habits and for better context-aware feedback, as discussed in Section 1.
    Ten healthy male participants (age range: 23–35) took part in our data collection experiments.
    However, not all activities were performed by each participant. Seven activities were performed
    by all ten participants, which are walking, jogging, biking, walking upstairs, walking downstairs,
    sitting and standing. These activities were performed for 3 min by each participant. Seven out of these
    ten participants performed eating, typing, writing, drinking coffee and giving a talk. These activities
    were performed for 5–6 min. Smoking was performed by six out of these ten participants, where
    each of them smoked one cigarette. Only six participants were smokers among the ten participants.
    We used 30 min of data for each activity with an equal amount of data from each participant. This
    resulted in a dataset of 390 (13x30) min.
	
    """

    def __init__(self, args):
        super(ComplexHA_HAR_DATA, self).__init__(args)
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """



        # the first cols no use
        self.used_cols = [1,2,3,4,5,6,7,8,9,10,11,12,13]  

        self.col_names =  ["accX", "accY", "accZ", "linX", "linY","linZ",
                           "gyrX", "gyrY", "gyrZ", "magX", "magY", "magZ", 
                           "activity_id"]

        """
        !!!!! No USER IDs !!! But the temporal relationship is perserved in the data. If We split the data into 10 folds. The test data
        has unseen data. reference from the paper
        ----------------------------------------------------
        For performance evaluation, we used 10-fold stratified cross-validation. In this validation method,
        the whole dataset is divided into ten equal parts or subsets. In each iteration, nine of these parts are
        used for training purpose and one for testing. This process is repeated ten times, thereby using all data
        for training, as well as testing. Stratified means that each fold or part has the right proportion of each
        class. Though we analyzed the classification accuracy, precision and F-measure as performance metrics,
        we only present the F-measure results, because it incorporates both accuracy and precision. Since we
        are only interested in the relative comparison of different scenarios, the F-measure as a performance
        metric is sufficient for this purpose. Moreover, our reported observations in the F-measure are similar
        to those in accuracy and precision. In Scikit-learn stratified cross-validation can be done in two ways:
        with shuffling and without shuffling the dataset.


        Without shuffling: In this method, no shuffling is performed before dividing the whole data into
        ten equal parts. The order of the data is preserved. In this way, the classification performance
        will be slightly lower than the shuffling method. In our case, it resembles a person-independent
        validation for the seven activities that were performed by all ten participants. 
        ! HERE !!!!!
        However, for the rest of the activities, it is not person independent. As the number of participants is less than 10,
        when we divide their data into ten equal parts, each part may contain data from more than one
        participant. This can lead to using data from one participant in both training and testing, with no
        overlap in data between training and testing sets. As the order of the time series data is preserved,
        the results are closer to the real-life situations.
        """
        self.train_keys   = [0,1,2,3,4,5,6,7,8]
        self.vali_keys    =[]
        self.test_keys    = [9]

        self.drop_activities = []
        
        self.file_encoding = {}
        
        self.label_map = [(11111, 'walk'), 
                          (11112, "stand"),
                          (11113, "jog"), 
                          (11114, "sit"), 
                          (11115, "bike"), 
                          (11116, "upstairs"), 
                          (11117, "downstairs"),
                          (11118, "type"),
                          (11119, "write"),
                          (11120, "coffee"),
                          (11121, "talk"),
                          (11122, "smoke"),
                          (11123, "eat")] 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.read_data()

    def load_the_data(self, root_path):

        pocket = pd.read_csv(os.path.join(root_path,"smartphoneatpocket.csv"), header=None)
        pocket =pocket.iloc[:,self.used_cols]
        pocket_cols = ["pocket_"+col for col in self.col_names]
        pocket.columns = pocket_cols


        wrist = pd.read_csv(os.path.join(root_path,"smartphoneatwrist.csv"), header=None)
        wrist = wrist.iloc[:,self.used_cols]
        wrist_cols = ["wrist_"+col for col in self.col_names]
        wrist.columns = wrist_cols

        assert (pocket["pocket_activity_id"] == wrist["wrist_activity_id"]).sum()   ==    wrist.shape[0]
        assert (pocket.index == wrist.index ).sum()    ==    wrist.shape[0]

        df_all = pd.concat([pocket,wrist], axis=1)

        del df_all["pocket_activity_id"]
        df_all["activity_id"] = df_all["wrist_activity_id"]
        del df_all["wrist_activity_id"]

        # check the size, it will be devided into 10 folds
        assert df_all.shape[0]/10 == 117000

        # add 10 sub_ids, it will be used for data split
        sub_id = []
        for i in range(10):
            sub_id.extend([i]*117000)
        df_all["sub_id"] = sub_id

        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # train_vali Test split 
        train_vali = pd.DataFrame()
        for key in self.train_keys + self.vali_keys:
            temp = df_all[df_all["sub_id"]==key]
            train_vali = pd.concat([train_vali,temp])

        test = pd.DataFrame()
        for key in self.test_keys:
            temp = df_all[df_all["sub_id"]==key]
            test = pd.concat([test,temp])



        # the col position varies between different datasets
        train_vali = train_vali.set_index('sub_id')
        train_vali_label = train_vali.iloc[:,-1]
        train_vali = train_vali.iloc[:,:-1]

        test = test.set_index('sub_id')
        test_label = test.iloc[:,-1]  
        test = test.iloc[:,:-1]

            
        return train_vali, train_vali_label, test, test_label