import numpy as np
import sys
import scipy.misc
import time
import os
from lib.models.data_providers import DataProvider
from lib.zero_shot import get_gap_ids

class TeapotsDataManager(object):
    def __init__(self, data_dir, batch_size, shuffle=True, gaps=True, file_ext='.npz',
                 train_fract=0.8, dev_fract=None, seed=None):
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gaps = gaps
        self.file_ext = file_ext.strip()
        self.train_fract = train_fract
        self.dev_fract = dev_fract
        seed = 123 if seed is None else seed
        self.rng = np.random.RandomState(seed)
        self.gap_ids = []
        self.inf_gen = False
                
        if self.file_ext == '.npz':
            self.__load = self.__load_npz
        else:
            self.__load = self.__load_imgs
            
        self.__load()
    
    def __set_data_splits(self):
        if self.dev_fract is None:
            self.dev_fract = round((1. - self.train_fract) / 2., 3)
        self.n_train = int(self.n_samples * self.train_fract)
        self.n_dev = int(self.n_samples * self.dev_fract)
        self.n_test = self.n_samples - (self.n_train + self.n_dev)
        print("Train set: {0}\nDev set: {1}\nTest set: {2}".format(self.n_train, self.n_dev, self.n_test))
                                     
    def __make_generator(self, imgs_dir, start_idx, end_idx):
        epoch_count = [0]
        def get_epoch():
            epoch_count[0] += 1
            if start_idx == 0: #Train 
                print("Epoch:{0}".format(epoch_count[0]))
            
            images = np.zeros((self.batch_size, 3, 64, 64), dtype='int32')
            files = list(range(start_idx, end_idx))
            if list(self.gap_ids):  #is not None
                files = [x for x in files if x not in self.gap_ids]

            if self.shuffle:
                self.rng.shuffle(files)

            for n, i in enumerate(files, 1): #start at 1, then % batch size == 0
                image = scipy.misc.imread(os.path.join(imgs_dir, ("im{0}" + self.file_ext).format(i)))
                images[n % self.batch_size] = image.transpose(2,0,1)
                if n > 0 and n % self.batch_size == 0:
                    yield (images,)
        
        return get_epoch

    def __load_imgs(self):
        img_dir = os.path.join(self.data_dir, 'images/')
        self.n_samples = len([name for name in os.listdir(img_dir)])
        self.__set_data_splits()
        if self.gaps:
            self.gap_ids = np.load(os.path.join(self.data_dir, 'gap_ids.npy'))
        self.train_gen = self.__make_generator(img_dir, 0, self.n_train)
        self.dev_gen   = self.__make_generator(img_dir, self.n_train, self.n_train + self.n_dev)
        self.test_gen  = self.__make_generator(img_dir, self.n_train + self.n_dev, self.n_train + self.n_dev + self.n_test)
    
    def __load_npz(self):
        # Load dataset
        data_zip = np.load(os.path.join(self.data_dir, "teapots.npz"))
        images = data_zip['images']
        gts = data_zip['gts']
        
        # Train, val, test splits
        self.n_samples = len(images)
        self.__set_data_splits()
        
        if self.gaps:
            self.gap_ids = get_gap_ids(gts)

        def split_data(start, end):
            imgs = np.delete(images[start:end], self.gap_ids, 0)
            gts = np.delete(gts[start:end], self.gap_ids, 0)
            return imgs, gts
              
        train_imgs, train_gts = split_data(0, self.n_train)
        dev_imgs, dev_gts = split_data(self.n_train, self.n_train + self.n_dev)
        test_imgs, test_gts = split_data(self.n_train + self.n_dev, self.n_train + self.n_dev + self.n_test)
        
        self.train_gen = DataProvider(train_imgs, train_gts, self.batch_size, shuffle_order=self.shuffle, rng=self.rng)
        self.dev_gen = DataProvider(dev_imgs, dev_gts, self.batch_size, shuffle_order=self.shuffle, rng=self.rng)
        self.test_gen = DataProvider(test_imgs, test_gts, self.batch_size, shuffle_order=self.shuffle, rng=self.rng)
    
    def inf_generators(self):
        def inf_gen(gen):
            while True:
                for (images,) in gen():
                    yield images
        self.train_gen = inf_gen(self.train_gen)
        self.dev_gen = inf_gen(self.dev_gen)
        self.test_gen = inf_gen(self.test_gen)
        self.inf_gen = True
                                       
    def get_generators(self):
        return self.train_gen, self.dev_gen, self.test_gen
    
    def set_divisor_batch_size(self):
        '''Ensure batch size evenly divides into n_samples.'''
        while self.n_samples % self.batch_size != 0:
            self.batch_size -= 1
    
    def __get_gen_data(self, g):
        samples = []
        if self.inf_gen:
            for n, batch in enumerate(g):
                samples.append(batch)
                if n % self.n_samples == 0:
                    break
        else:
            for (batch,) in g():
                samples.append(batch)
        np.vstack(samples)
        
    def get_train_data(self):
        return self.__get_gen_data(self.train_gen)
    
    def get_dev_data(self):
        return self.__get_gen_data(self.dev_gen)
    
    def get_test_data(self):
        return self.__get_gen_data(self.test_gen)
    
    def get_all_samples(self):
        samples = []
        if self.train_fract  == 1.:
            return self.__get_gen_data(self.train_gen)
        
        tr_f = self.train_fract
        dev_f = self.dev_fract
        test_f = 1. - (self.train_fract + self.dev_fract)
        for fr, g in zip([tr_f, dev_f, test_f], [self.train_gen, self.dev_gen, self.test_gen]):
            if fract <= 0.:
                break
            samples.append(self.__get_gen_data(g))
        return np.vstack(samples)