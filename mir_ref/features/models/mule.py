"""Code used for inference for MULE model.
Source: https://github.com/PandoraMedia/music-audio-features:
"""

import tempfile

import numpy as np
from scooch import ConfigList, Configurable, Param


class Feature(Configurable):
    """
    The base class for all feature types.
    """

    def __del__(self):
        """
        **Destructor**
        """
        if hasattr(self, "_data_file"):
            self._data_file.close()

    def add_data(self, data):
        """
        Adds data to extend the object's current data via concatenation along the time axis.
        This is useful for populating data in chunks, where populating it all at once would
        cause excessive memory usage.

        Arg:
            data: np.ndarray - The data to be appended to the object's memmapped numpy array.
        """
        if not hasattr(self, "_data_file"):
            self._data_file = tempfile.NamedTemporaryFile(mode="w+b")

        if not hasattr(self, "_data") or self._data is None:
            original_data_size = 0
        else:
            original_data_size = self._data.shape[1]
        final_size = original_data_size + data.shape[1]

        filename = self._data_file.name

        self._data = np.memmap(
            filename,
            dtype="float32",
            mode="r+",
            shape=(data.shape[0], final_size),
            order="F",
        )
        self._data[:, original_data_size:] = data

    def _extract(self, source, length):
        """
        Extracts feature data file or other feature a given time-chunk.

        Args:
            source_feature: mule.features.Feature - The feature to transform.

            start_time: int - The index in the input at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        raise NotImplementedError(
            f"The {self.__name__.__class__} has no feature extraction method"
        )

    def save(self, path):
        """
        Save the feature data blob to disk.

        Args:
            path: str - The path to save the data to.
        """
        np.save(path, self.data)

    def clear(self):
        """
        Clears any previously analyzed feature data, ready for a new analysis.
        """
        self._data = None
        if hasattr(self, "_data_file"):
            self._data_file.close()
            del self._data_file

    @property
    def data(self):
        """
        The feature data blob itself.
        """
        return self._data


class SourceFile(Configurable):
    """
    Base class for SCOOCH configurable file readers.
    """

    def load(self, fname):
        """
        Any preprocessing steps to load a file prior to reading it.

        Args:
            fname: file-like - A file like object to be loaded.
        """
        raise NotImplementedError(
            f"The class, {self.__class__.__name__}, has no method for loading files"
        )

    def read(self, n):
        """
        Reads an amount of data from the file.

        Args:
            n: int - A size parameter indicating the amount of data to read.

        Return:
            object - The decoded data read and in memory.
        """
        raise NotImplementedError(
            f"The class, {self.__class__.__name__}, has no method for reading files"
        )

    def close(self):
        """
        Closes any previously loaded file.
        """
        raise NotImplementedError(
            f"The class, {self.__class__.__name__}, has no method for closing files"
        )

    def __len__(self):
        raise NotImplementedError(
            f"The class, {self.__class__.__name__} has no method for determining file data length"
        )


class SourceFeature(Feature):
    """
    A feature that is derived directly from raw data, e.g., a data file.
    """

    # SCOOCH Configuration
    _input_file = Param(
        SourceFile,
        doc="The file object defining the parameters of the raw data that this feature is constructed from.",
    )

    _CHUNK_SIZE = 44100 * 60 * 15

    # Methods
    def from_file(self, fname):
        """
        Takes a file and processes its data in chunks to form a feature.

        Args:
            fname: str - The path to the input file from which this feature is constructed.
        """
        # Load file
        self._input_file.load(fname)

        # Read samples into data
        processed_input_frames = 0
        while processed_input_frames < len(self._input_file):
            data = self._extract(
                self._input_file, processed_input_frames, self._CHUNK_SIZE
            )
            processed_input_frames += self._CHUNK_SIZE
            self.add_data(data)

    def clear(self):
        """
        Clears any previously analyzed feature data, ready for a new analysis.
        """
        super().clear()
        self._input_file.close()

    def __len__(self):
        """
        Returns the number of bytes / samples / indices in the input data file.
        """
        return len(self._input_file)


class Extractor(Configurable):
    """
    Base class for classes that are responsible for extracting data
    from mule.features.Feature classes.
    """

    #
    # Methods
    #
    def extract_range(self, feature, start_index, end_index):
        """
        Extracts data over a given index range from a single feature.

        Args:
            feature: mule.feature.Feature - A feature to extract data from.

            start_index: int - The first index (inclusive) at which to return data.

            end_index: int - The last index (exclusive) at which to return data.

        Return:
            numpy.ndarray - The extracted feature data. Features on first axis, time on
            second axis.
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} class has no `extract_range` method."
        )

    def extract_batch(self, features, indices):
        """
        Extracts a batch of features from potentially multiple features, each potentially
        at distinct indices.

        Args:
            features: list(mule.features.Feature) - A list of features from which to extract
            data from.

            indices: list(int) - A list of indices, the same size as `features`. Each element
            provides an index at which to extract data from the coressponding element in the
            `features` argument.

        Return:
            np.ndarray - A batch of features, with features on the batch dimension on the first
            axis and feature data on the remaining axes.
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} class has no `extract_batch` method."
        )


class TransformFeature(Feature):
    """
    Base class for all features that are transforms of other features.
    """

    #
    # SCOOCH Configuration
    #
    _extractor = Param(
        Extractor,
        doc="An object defining how data will be extracted from the input feature and provided to the transformation of this feature.",
    )

    # The size in time of each chunk that this feature will process at any one time.
    _CHUNK_SIZE = 44100 * 60 * 15

    #
    # Methods
    #
    def from_feature(self, source_feature):
        """
        Populates this features data as a transform of the provided input feature.

        Args:
            source_feature: mule.features.Feature - A feature from which this feature will
            be created as a transformation thereof.
        """
        boundaries = list(range(0, len(source_feature), self._CHUNK_SIZE)) + [
            len(source_feature)
        ]
        chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        for start_time, end_time in chunks:
            data = self._extract(source_feature, start_time, end_time - start_time)
            if data is not None and len(data):
                self.add_data(data)

    def _extract(self, source_feature, start_time, chunk_size):
        """
        Extracts feature data as a transformation of a given source feature for a given
        time-chunk.

        Args:
            source_feature: mule.features.Feature - The feature to transform.

            start_time: int - The index in the feature at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        end_time = start_time + chunk_size
        return self._extractor.extract_range(source_feature, start_time, end_time)

    def __len__(self):
        if hasattr(self, "_data"):
            return self._data.shape[1]
        else:
            return 0


class Analysis(Configurable):
    """
    A class encapsulating analysis of a single input file.
    """

    # SCOOCH Configuration
    _source_feature = Param(
        SourceFeature, doc="The feature used to decode the provided raw file data."
    )
    _feature_transforms = Param(
        ConfigList(TransformFeature),
        doc="Feature transformations to apply, in order, to the source feature generated from the input file.",
    )

    # Methods
    def analyze(self, fname):
        """
        Analyze features for a single filepath.

        Args:
            fname: str - The filename path, from which to generate features.

        Return:
            mule.features.Feature - The feature resulting from the configured feature
            transformations.
        """
        for feat in [self._source_feature] + self._feature_transforms:
            feat.clear()

        self._source_feature.from_file(fname)
        input_feature = self._source_feature
        for feature in self._feature_transforms:
            feature.from_feature(input_feature)
            input_feature = feature

        return input_feature
