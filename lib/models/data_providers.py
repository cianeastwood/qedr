# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import numpy as np
from scipy.misc import imread


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(123)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()

class FlexibleDataProvider(DataProvider):
    '''
    Data provider with added flexibility/functionality:
    1) Infinite iterations possible (optional raising of StopIteration())
    2) Unsupervised training (optional targets)
    3) Print epoch
    '''
    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 inf=False, shuffle_order=True, rng=None, print_epoch=False):
        self.inf = inf
        self.print_epoch = print_epoch
        self.epoch = 0
        super(FlexibleDataProvider, self).__init__(inputs, targets, 
              batch_size, max_num_batches, shuffle_order, rng)
    
    def new_epoch(self):
        super(FlexibleDataProvider, self).new_epoch()
        self.epoch += 1
        if self.print_epoch:
            print("Epoch:{0}".format(self.epoch))
    
    def reset(self):
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        if self.targets is not None:
            self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        if self.targets is not None:
            self.targets = self.targets[perm]
    
    def next(self):
        if self._curr_batch + 1 > self.num_batches:
            self.new_epoch()
            if not self.inf:
                raise StopIteration()
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        if self.targets is not None:
            targets_batch = self.targets[batch_slice]
        else:
            targets_batch = None
        self._curr_batch += 1
        return inputs_batch, targets_batch    

class FlexibleImageDataProvider(FlexibleDataProvider):
    """
    FlexbileDataProvider which reads batch data directly from .jpeg, .png, etc. 
    files rather than an input array. Filenames: im[file_id].jpeg/png/etc.
    
    inputs: int array of file_ids in range [0, n_samples]
    """
 
    def __init__(self, imgs_dir, inputs, targets, batch_size, image_shape,
                 file_ext='.jpeg', inf=False, shuffle_order=True, gap_ids=[],
                 rng=None, print_epoch=False, max_num_batches=-1, dtype='int32'):

        self.imgs_dir = imgs_dir
        self.image_shape = image_shape
        self.file_ext = file_ext
        self.dtype = dtype
        super(FlexibleImageDataProvider, self).__init__(inputs, targets, 
              batch_size, max_num_batches, inf, shuffle_order, rng, print_epoch)
    
    def _read_images(self, batch_file_ids):       
        images = np.zeros([self.batch_size] + self.image_shape, dtype=self.dtype)
        for n, b_id in enumerate(batch_file_ids):
            image = imread(os.path.join(self.imgs_dir, ("im{0}" + self.file_ext).format(b_id)))
            if list(image.shape) != self.image_shape:
                if list(image.transpose(2,0,1).shape) == self.image_shape: # e.g. (64,64,3)->(3,64,64)
                    image = image.transpose(2,0,1)
                else:
                    raise Exception("Image does not match specified shape.")
            images[n % self.batch_size] = image  
        return images
        
    def next(self):
        inputs_batch, targets_batch = super(FlexibleImageDataProvider, self).next() # inputs = file_ids
        return self._read_images(inputs_batch), targets_batch
    
    
class TeapotsDataProvider(FlexibleDataProvider):    
    
    def next(self):
        inputs_batch, targets_batch = super(TeapotsDataProvider, self).next()
        inputs_batch = inputs_batch.transpose(0,3,1,2) #(-1,64,64,3)->(-1,3,64,64)
        return inputs_batch, targets_batch
    

# class TeapotsImageDataProvider(FlexibleDataProvider):    
 
#     def __init__(self, imgs_dir, inputs, targets, batch_size, image_shape,
#                  file_ext='.jpeg', inf=False, shuffle_order=True, gap_ids=[],
#                  rng=None, print_epoch=False, max_num_batches=-1):

#         self.imgs_dir = imgs_dir
#         self.image_shape = image_shape
#         self.file_ext = file_ext
#         super(TeapotsBatchDataProvider, self).__init__(inputs, targets, 
#               batch_size, max_num_batches, inf, shuffle_order, rng, print_epoch)
        
#     def _make_generator(self):
#         def get_epoch():            
#             images = np.zeros([self.batch_size] + self.image_shape), dtype='int32')
#             for n, i in enumerate(self.inputs, 1): #start at 1, then % batch size == 0
#                 image = scipy.misc.imread(os.path.join(self.imgs_dir, ("im{0}" + self.file_ext).format(i)))
#                 images[n % self.batch_size] = image.transpose(2,0,1) # (64,64,3)->(3,64,64)
#                 if n > 0 and n % self.batch_size == 0:
#                     yield (images,)
        
#         return get_epoch

#     def new_epoch(self):
#         self._curr_batch = 0
#         if self.shuffle_order:
#             self.shuffle()
#         self.gen = self._make_generator()
#         self.epoch += 1
#         if self.print_epoch:
#             print("Epoch:{0}".format(self.epoch))
        
#     def next(self):
#         if self._curr_batch + 1 > self.num_batches:
#             self.new_epoch()
#             if not self.inf:
#                 raise StopIteration()
#         self._curr_batch += 1
#         return next(self.gen())