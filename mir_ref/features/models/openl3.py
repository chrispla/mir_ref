"""Code for OpenL3 embedding model inference.
Source: https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8
"""

from pathlib import Path
import essentia.standard as es
import numpy as np
from essentia import Pool


class MelSpectrogramOpenL3:
    def __init__(self, hop_time):
        self.hop_time = hop_time

        self.sr = 48000
        self.n_mels = 128
        self.frame_size = 2048
        self.hop_size = 242
        self.a_min = 1e-10
        self.d_range = 80
        self.db_ref = 1.0

        self.patch_samples = int(1 * self.sr)
        self.hop_samples = int(self.hop_time * self.sr)

        self.w = es.Windowing(
            size=self.frame_size,
            normalized=False,
        )
        self.s = es.Spectrum(size=self.frame_size)
        self.mb = es.MelBands(
            highFrequencyBound=self.sr / 2,
            inputSize=self.frame_size // 2 + 1,
            log=False,
            lowFrequencyBound=0,
            normalize="unit_tri",
            numberBands=self.n_mels,
            sampleRate=self.sr,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute(self, audio_file):
        audio = es.MonoLoader(filename=audio_file, sampleRate=self.sr)()

        batch = []
        for audio_chunk in es.FrameGenerator(
            audio, frameSize=self.patch_samples, hopSize=self.hop_samples
        ):
            melbands = np.array(
                [
                    self.mb(self.s(self.w(frame)))
                    for frame in es.FrameGenerator(
                        audio_chunk,
                        frameSize=self.frame_size,
                        hopSize=self.hop_size,
                        validFrameThresholdRatio=0.5,
                    )
                ]
            )

            melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
            melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
            melbands = np.maximum(melbands, melbands.max() - self.d_range)
            melbands -= np.max(melbands)

            batch.append(melbands.copy())

        return np.vstack(batch)


class EmbeddingsOpenL3:
    def __init__(self, graph_path, hop_time=1, batch_size=60, melbands=128):
        self.hop_time = hop_time
        self.batch_size = batch_size

        self.graph_path = Path(graph_path)

        self.x_size = 199
        self.y_size = melbands
        self.squeeze = False

        self.permutation = [0, 3, 2, 1]

        self.input_layer = "melspectrogram"
        self.output_layer = "embeddings"

        self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time)

        self.model = es.TensorflowPredict(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer],
            squeeze=self.squeeze,
        )

    def compute(self, audio_file):
        mel_spectrogram = self.mel_extractor.compute(audio_file)
        # in OpenL3 the hop size is computed in the feature extraction level

        hop_size_samples = self.x_size

        batch = self.__melspectrogram_to_batch(mel_spectrogram, hop_size_samples)

        pool = Pool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)
            pool.set(self.input_layer, batch[start:end])
            out_pool = self.model(pool)
            embeddings.append(out_pool[self.output_layer].squeeze())

        return np.vstack(embeddings)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        batch = es.TensorTranspose(permutation=self.permutation)(batch)

        return batch
