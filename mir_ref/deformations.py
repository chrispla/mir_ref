"""Implementations of various audio deformations used for
robustness evaluation.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from pathlib import Path

import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm


def deform_audio(
    track_id: str,
    dataset,
):
    """For each deformation scenario, deform and save single audio file."""

    # load audio
    y, sr = librosa.load(dataset.audio_paths[track_id], sr=None)

    for scenario_idx, scenario in enumerate(dataset.deformations_cfg):
        y_d = y.copy()
        # we're looping like this so that we retain the order of
        # deformations provided by the user
        # debatable whether this should be the default behavior
        for deformation in scenario:
            if deformation["type"] == "AddGaussianSNR":
                from audiomentations import AddGaussianSNR

                transform = AddGaussianSNR(**deformation["params"])
            elif deformation["type"] == "ApplyImpulseResponse":
                from audiomentations import ApplyImpulseResponse

                transform = ApplyImpulseResponse(**deformation["params"])
            elif deformation["type"] == "ClippingDistortion":
                from audiomentations import ClippingDistortion

                transform = ClippingDistortion(**deformation["params"])
            elif deformation["type"] == "Gain":
                from audiomentations import Gain

                transform = Gain(**deformation["params"])
            elif deformation["type"] == "Mp3Compression":
                from audiomentations import Mp3Compression

                transform = Mp3Compression(**deformation["params"])
            elif deformation["type"] == "PitchShift":
                from audiomentations import PitchShift

                transform = PitchShift(**deformation["params"])
            else:
                raise ValueError(f"Deformation {deformation['type']} not implemented.")

            # apply deformation
            y_d = transform(y_d, sr)
            del transform

        # save deformed audio
        output_filepath = dataset.get_deformed_audio_path(
            track_id=track_id, deform_idx=scenario_idx
        )

        # create parent dirs if they don't exist, and write audio
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_filepath, y_d, sr)


def deform_audio_essentia(
    track_id: str,
    dataset,
):
    """Some of the essentia ffmpeg calls seem to have deprecated parameters,
    something related to the time_base used. I'm temporarily moving the
    essentia loading and writing here.
    """
    # move to imports if used
    from essentia.standard import AudioLoader, AudioWriter

    input_filepath = dataset.audio_paths[track_id]

    # load audio
    y, sr, channels, _, bit_rate, _ = AudioLoader(filename=input_filepath)()
    # assert audio is mono or stereo
    assert channels <= 2
    # some book keeping for constructing the output path later
    file_dir = Path(input_filepath).parent
    file_stem = str(Path(input_filepath).stem)
    file_suffix = str(Path(input_filepath).suffix)

    for scenario_idx, scenario in enumerate(dataset.deformations_cfg):
        y_d = y.copy()
        # we're looping like this so that we retain the order of
        # deformations provided by the user
        # debatable whether this should be the default behavior
        for deformation in scenario:
            if deformation["type"] == "AddGaussianSNR":
                from audiomentations import AddGaussianSNR

                transform = AddGaussianSNR(**deformation["params"])
            elif deformation["type"] == "ApplyImpulseResponse":
                from audiomentations import ApplyImpulseResponse

                transform = ApplyImpulseResponse(**deformation["params"])
            elif deformation["type"] == "ClippingDistortion":
                from audiomentations import ClippingDistortion

                transform = ClippingDistortion(**deformation["params"])
            elif deformation["type"] == "Gain":
                from audiomentations import Gain

                transform == Gain(**deformation["params"])
            elif deformation["type"] == "Mp3Compression":
                from audiomentations import Mp3Compression

                transform = Mp3Compression(**deformation["params"])
            elif deformation["type"] == "PitchShift":
                from audiomentations import PitchShift

                transform = PitchShift(**deformation["params"])
            else:
                raise ValueError(f"Deformation {deformation['type']} not implemented.")

            # apply deformation
            y_d = transform(y_d, sr)

        # save deformed audio
        output_filepath = dataset.get_deformed_audio_path(
            track_id=track_id, deform_idx=scenario_idx
        )

        # create parent dirs if they don't exist, and write audio
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

        # special case for lossy compression formats in which bitrate needs to be specified
        lossy_format_bitrates = [
            32,
            40,
            48,
            56,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            192,
            224,
            256,
            320,
        ]
        if (
            file_suffix == ".mp3" or file_suffix == ".ogg"
        ) and bit_rate in lossy_format_bitrates:
            AudioWriter(
                filename=output_filepath,
                format=file_suffix[1:],
                sampleRate=sr,
                bitrate=bit_rate,
            )(y_d)
        else:
            AudioWriter(
                filename=output_filepath, format=file_suffix[1:], sampleRate=sr
            )(y_d)


def generate_deformations(
    dataset,
    n_jobs: int = 1,
):
    """Generate deformed audio and save."""

    # check if there are no deformations specified in the experiment
    if not dataset.deformations_cfg:
        print(
            f"No deformations specified for '{dataset.name}'. Skipping deformation generation."
        )
        return

    # create output dir for deformed audio if it doesn't exist
    (Path(dataset.data_dir) / "audio_deformed").mkdir(parents=True, exist_ok=True)

    if n_jobs == 1:
        for track_id in tqdm(dataset.track_ids):
            deform_audio(track_id, dataset)
    else:
        # this passes around the dataset object, which is not ideal for performance
        Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(deform_audio)(track_id, dataset)
            for track_id in tqdm(dataset.track_ids)
        )

    print("Deformed audio generated and saved.")
