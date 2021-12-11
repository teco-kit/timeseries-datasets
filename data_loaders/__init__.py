from torch.utils.data import Dataset
import numpy as np
from .dataloader_uci_har import UCI_HAR_DATA
from .dataloader_pamap2_har import PAMAP2_HAR_DATA
from .dataloader_opportunity_har import Opportunity_HAR_DATA
from .dataloader_daphnet_har import Daphnet_HAR_DATA
from .dataloader_usc_had_har import USC_HAD_HAR_DATA
from .dataloader_skoda_har import Skoda_HAR_DATA
from .dataloader_mhealth_har import Mhealth_HAR_DATA
from .dataloader_wisdm_har import WISDM_HAR_DATA
from .dataloader_fusion_har import FUSION_HAR_DATA
from .dataloader_dsa_har import DSA_HAR_DATA
from .dataloader_single_chest_har import Single_Chest_HAR_DATA
from .dataloader_utd_mhad_har import UTD_MHAD_HAR_DATA

data_dict = {"ucihar" : UCI_HAR_DATA,
             "pamap2" : PAMAP2_HAR_DATA,
             "opportunity":Opportunity_HAR_DATA,
             "daphnet":Daphnet_HAR_DATA,
             "uschad": USC_HAD_HAR_DATA,
             "skoda": Skoda_HAR_DATA,
             "mhealth": Mhealth_HAR_DATA,
             "wisdm" : WISDM_HAR_DATA,
             "fusion" : FUSION_HAR_DATA,
             "dsa" : DSA_HAR_DATA,
             "single_chest" : Single_Chest_HAR_DATA,
             "utd_mhad" : UTD_MHAD_HAR_DATA}






class data_set(Dataset):
    def __init__(self, args, dataset, flag):
        """
        args : a dict , In addition to the parameters for building the model, the parameters for reading the data are also in here
        dataset : It should be implmented dataset object, it contarins train_x, train_y, vali_x,vali_y,test_x,test_y
        flag : (str) "train","test","vali"
        """
        self.args = args
        self.flag = flag



        if self.flag == "train":
            # load train
            self.data_x = dataset.train_vali_x.copy()
            self.data_y = dataset.train_vali_y.copy()
            self.window_index =  dataset.train_window_index
            print("Train data number : ", len(self.window_index))


        elif self.flag == "vali":
            # load vali
            self.data_x = dataset.train_vali_x.copy()
            self.data_y = dataset.train_vali_y.copy()
            self.window_index =  dataset.vali_window_index
            print("Validation data number : ",  len(self.window_index))  


        else:
            # load test
            self.data_x = dataset.test_x.copy()
            self.data_y = dataset.test_y.copy()
            self.window_index = dataset.test_window_index
            print("Test data number : ", len(self.window_index))  
            
            
        all_labels = list(np.unique(np.concatenate((dataset.train_vali_y, dataset.test_y), axis=0)))
        to_drop = list(dataset.drop_activities)
        label = [item for item in all_labels if item not in to_drop]

        self.nb_classes = len(label)
        assert self.nb_classes==len(dataset.no_drop_activites)

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}
        self.class_back_transform = {i: x for i, x in enumerate(classes)}
        self.input_length = self.window_index[0][1]-self.window_index[0][0]
        self.channel_in = self.data_x.shape[1]

        if self.flag == "train":
            print("The number of classes is : ", self.nb_classes)
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)


    def __getitem__(self, index):
        start_index = self.window_index[index][0]
        end_index = self.window_index[index][1]
        sample_x = self.data_x.iloc[start_index:end_index].values
        sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

        if self.args.spectrogram:
            # TODO need to be implemented
            sample_x = self.spec_list[index]
            
        return sample_x, sample_y

    def __len__(self):
        return len(self.window_index)

