"""Dataloaders.
Code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

from pathlib import Path

import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
        self,
        ids_list,
        labels_dict,
        paths_dict,
        dim,
        n_classes,
        batch_size=8,
        shuffle=True,
    ):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.ids_list = ids_list
        self.labels_dict = labels_dict
        self.paths_dict = paths_dict
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch. Last non-full batch is dropped."""
        return int(np.floor(len(self.ids_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of ids
        ids_list_temp = [self.ids_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(ids_list_temp=ids_list_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_list_temp):
        """Generates data containing batch_size samples
        X : (n_samples, *dim)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, t_id in enumerate(ids_list_temp):
            # Store sample
            emb = np.load(self.paths_dict[t_id])
            X[i,] = np.reshape(emb, *self.dim)

            y[i] = self.labels_dict[t_id]

        return X, y
