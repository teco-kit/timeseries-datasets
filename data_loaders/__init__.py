
__all__ = [
    "ucihar_data_set",
    "pamap2_data_set"
]

from torch.utils.data import Dataset
import numpy as np
from .dataloader_uci_har import UCI_HAR_DATA
from .dataloader_pamap2_har import PAMAP2_HAR_DATA
from .dataloader_opportunity_har import Opportunity_HAR_DATA
from .dataloader_daphnet_har import Daphnet_HAR_DATA


data_dict = {"ucihar" : UCI_HAR_DATA,
             "pamap2" : PAMAP2_HAR_DATA,
             "opportunity":Opportunity_HAR_DATA,
             "daphnet":Daphnet_HAR_DATA}






class ucihar_data_set(Dataset):
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
            self.data_x = dataset.train_x.copy()
            self.data_y = dataset.train_y.copy()
            print("Train data number : ", len(self.data_x)/self.data_x.loc[0].shape[0])        
        elif self.flag == "vali":
            # load vali
            self.data_x = dataset.vali_x.copy()
            self.data_y = dataset.vali_y.copy()
            print("Validation data number : ", len(self.data_x)/self.data_x.loc[0].shape[0])   
        else:
            # load test
            self.data_x = dataset.test_x.copy()
            self.data_y = dataset.test_y.copy()
            print("Test data number : ", len(self.data_x)/self.data_x.loc[0].shape[0])  
            
            
        self.nb_classes = len(np.unique(np.concatenate((dataset.train_y, dataset.vali_y, dataset.test_y), axis=0)))
        self.input_length = self.data_x.loc[0].shape[0]
        self.channel_in = self.data_x.loc[0].shape[1]

        if self.flag == "train":
            print("The number of classes is : ", self.nb_classes)
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)


    def __getitem__(self, index):
        sample_x = self.data_x.loc[index].values
        sample_y = self.data_y[index]

        if self.args.spectrogram:
            sample_x = self.spec_list[index]
            
        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x.index.unique())


class opportunity_data_set(Dataset):
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
        sample_y = self.data_y.iloc[start_index:end_index].mode().loc[0]

        if self.args.spectrogram:
            # TODO need to be implemented
            sample_x = self.spec_list[index]
            
        return sample_x, sample_y

    def __len__(self):
        return len(self.window_index)



class pamap2_data_set(Dataset):
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
        sample_y = self.data_y.iloc[start_index:end_index].mode().loc[0]

        if self.args.spectrogram:
            # TODO need to be implemented
            sample_x = self.spec_list[index]
            
        return sample_x, sample_y

    def __len__(self):
        return len(self.window_index)

class daphnet_data_set(Dataset):
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
        sample_y = self.data_y.iloc[start_index:end_index].mode().loc[0]

        if self.args.spectrogram:
            # TODO need to be implemented
            sample_x = self.spec_list[index]
            
        return sample_x, sample_y

    def __len__(self):
        return len(self.window_index)

dataloader_dict = {"ucihar" : ucihar_data_set,
                   "pamap2" : pamap2_data_set,
                   "opportunity":opportunity_data_set,
                   "daphnet":daphnet_data_set}