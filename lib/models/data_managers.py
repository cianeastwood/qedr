import numpy as np
import sys
import scipy.misc
import time
import os
from lib.models.data_providers import FlexibleDataProvider, FlexibleImageDataProvider
from lib.models.data_providers import TeapotsDataProvider
from lib.zero_shot import get_gap_ids

class DataManager(object):
    def __init__(self, data_dir, dataset_name, batch_size, image_shape, 
                 shuffle=True, gaps=False, file_ext='.npz', train_fract=0.8, 
                 dev_fract=None, inf=True, supervised=False):
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.gaps = gaps
        self.file_ext = file_ext.strip()
        self.train_fract = train_fract
        self.dev_fract = dev_fract
        self.inf = inf
        self.supervised = supervised
        self.gap_ids = []
        
        if self.file_ext == '.npz':
            self._data_provider = FlexibleDataProvider
            self.__create_data_provider = self.__create_data_provider_npz
        else:
            self._data_provider = FlexibleImageDataProvider
            self.__create_data_provider = self.__create_data_provider_imgs
    
    def __set_data_splits(self):
        if self.dev_fract is None:
            self.dev_fract = round((1. - self.train_fract) / 2., 3)
        self.n_train = int(self.n_samples * self.train_fract)
        self.n_dev = int(self.n_samples * self.dev_fract)
        self.n_test = self.n_samples - (self.n_train + self.n_dev)
        print("Train set: {0}\nDev set: {1}\nTest set: {2}".format(
              self.n_train, self.n_dev, self.n_test))
                                     
    def __split_data(self, data, start_idx, end_idx):
        return np.delete(data[start_idx:end_idx], self.gap_ids, 0)
    
    def __create_data_provider_imgs(self):
        img_dir = os.path.join(self.data_dir, 'images/')
        self.n_samples = len([name for name in os.listdir(img_dir)])
        img_files = np.array(list(range(0, self.n_samples))) #file ids [0, n_samples-1]
        self.__set_data_splits()
        
        if self.gaps:
            self.gap_ids = np.load(os.path.join(self.data_dir, 'gap_ids.npy'))
        
        train_img_ids = self.__split_data(img_files, 0, 
                                          self.n_train) #file ids
        dev_img_ids  = self.__split_data(img_files, self.n_train, 
                                          self.n_train + self.n_dev)
        test_img_ids = self.__split_data(img_files, self.n_train + self.n_dev, 
                                          self.n_train + self.n_dev + self.n_test)
        
        if self.supervised:
            gts = np.load(os.path.join(self.data_dir, self.dataset_name + ".npz"))['gts']
            train_gts = self.__split_data(gts, 0, 
                                          self.n_train)
            dev_gts   = self.__split_data(gts, self.n_train, 
                                          self.n_train + self.n_dev)
            test_gts  = self.__split_data(gts, self.n_train + self.n_dev, 
                                          self.n_train + self.n_dev + self.n_test)
        else:
            train_gts, dev_gts, test_gts = None, None, None
        
        self.train = self._data_provider(img_dir, train_img_ids, train_gts, 
                                        self.batch_size, self.image_shape,
                                        self.file_ext, self.inf, self.shuffle, 
                                        print_epoch=True)
        self.dev   = self._data_provider(img_dir, dev_img_ids, dev_gts, 
                                        self.batch_size, self.image_shape,
                                        self.file_ext, self.inf, self.shuffle)
        self.test  = self._data_provider(img_dir, test_img_ids, test_gts, 
                                        self.batch_size, self.image_shape,
                                        self.file_ext, self.inf, self.shuffle)
    
    def __create_data_provider_npz(self):
        data_zip = np.load(os.path.join(self.data_dir, self.dataset_name + ".npz"))
        images = data_zip['images']
        self.n_samples = len(images)
        self.__set_data_splits()
        
        if self.gaps:
            self.gap_ids = np.load(os.path.join(self.data_dir, 'gap_ids.npy'))
              
        train_imgs = self.__split_data(images, 0, 
                                       self.n_train)
        dev_imgs   = self.__split_data(images, self.n_train, 
                                       self.n_train + self.n_dev)
        test_imgs  = self.__split_data(images, self.n_train + self.n_dev, 
                                       self.n_train + self.n_dev + self.n_test)
        
        if self.supervised:
            gts = data_zip['gts']
            train_gts  = self.__split_data(gts, 0, 
                                           self.n_train)
            dev_gts    = self.__split_data(gts, self.n_train, 
                                           self.n_train + self.n_dev)
            test_gts   = self.__split_data(gts, self.n_train + self.n_dev, 
                                           self.n_train + self.n_dev + self.n_test)
        else:
            train_gts, dev_gts, test_gts = None, None, None
        
        self.train = self._data_provider(train_imgs, train_gts, self.batch_size, 
                                          inf=self.inf, shuffle_order=self.shuffle, 
                                          print_epoch=True)
        self.dev   = self._data_provider(dev_imgs, dev_gts, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)
        self.test  = self._data_provider(test_imgs, test_gts, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)
                                       
    def get_iterators(self):
        self.__create_data_provider()
        return self.train, self.dev, self.test
    
    def set_divisor_batch_size(self):
        '''Ensure batch size evenly divides into n_samples.'''
        while self.n_samples % self.batch_size != 0:
            self.batch_size -= 1

            
class TeapotsDataManager(DataManager):
    def __init__(self, data_dir, batch_size, image_shape, shuffle=True, 
                 gaps=True, file_ext='.npz', train_fract=0.8, 
                 dev_fract=None, inf=True, supervised=False):
        
        super(TeapotsDataManager, self).__init__(data_dir, "teapots", 
              batch_size, image_shape, shuffle, gaps, file_ext,
              train_fract, dev_fract, inf, supervised)
        
        if self.file_ext == '.npz':
            self._data_provider = TeapotsDataProvider #transpose image batch in provider