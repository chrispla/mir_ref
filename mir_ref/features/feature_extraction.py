"""Retrieve correct feature extractor and extract features.
"""

import os

import wget

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from tqdm import tqdm


def check_model_exists(model_path: str):
    if not os.path.exists(model_path):
        raise Exception(
            f"Model not found at {model_path}"
            + "Please download it from https://essentia.upf.edu/models.html"
            + "and place it in the mir_ref/features/models/weights directory."
        )


def get_input_output_paths(
    dataset,
    model_name,
    skip_clean=False,
    skip_deformed=False,
    no_overwrite=False,
    deform_list=None,
):
    """Get a list of input audio paths (including deformed audio)
    and a list of the corresponding output paths for the embeddings
    for a given dataset and embedding model. Don't include embeddings
    that have already been computed.

    Args:
        dataset: mir_ref Dataset object.
        model_name: Name of the embedding model.
        skip_clean: Whether to skip embedding generation for clean audio.
        skip_deformed: Whether to skip embedding generation for deformed audio.
        no_overwrite: Whether to skip embedding generation for existing embeddings.
        deform_list: List of deformation scenario indicies to include. If None,
                     include all deformation scenarios.
    """

    # get audio paths for clean and deformed audio
    if not skip_clean:
        audio_paths = [dataset.audio_paths[track_id] for track_id in dataset.track_ids]
    else:
        audio_paths = []
    if not skip_deformed and dataset.deformations_cfg is not None:
        if deform_list is None:
            deform_list = range(len(dataset.deformations_cfg))
        audio_paths += [
            dataset.get_deformed_audio_path(track_id=track_id, deform_idx=deform_idx)
            for deform_idx in deform_list
            for track_id in dataset.track_ids
        ]

    # get output embedding paths for the same tracks
    if not skip_clean:
        emb_paths = [
            dataset.get_embedding_path(track_id, model_name)
            for track_id in dataset.track_ids
        ]
    else:
        emb_paths = []
    if not skip_deformed and dataset.deformations_cfg is not None:
        if deform_list is None:
            deform_list = range(len(dataset.deformations_cfg))
        emb_paths += [
            dataset.get_deformed_embedding_path(
                track_id=track_id, feature=model_name, deform_idx=deform_idx
            )
            for deform_idx in deform_list
            for track_id in dataset.track_ids
        ]

    if no_overwrite:
        # remove audio paths and the respective embeddings paths if embedding
        # already exists
        audio_paths, emb_paths = zip(
            *[
                (audio_path, emb_path)
                for audio_path, emb_path in zip(audio_paths, emb_paths)
                if not os.path.exists(emb_path)
            ]
        )

    return audio_paths, emb_paths


def compute_and_save_embeddings(
    model: object,
    model_name: str,
    aggregation: str,
    dataset,
    sample_rate: int,
    resample_quality=1,
    skip_clean=False,
    skip_deformed=False,
    no_overwrite=False,
    deform_list=None,
):
    """Compute embeddings given model object and
    audio path list.
    """

    from essentia.standard import MonoLoader

    monoloader = MonoLoader(sampleRate=sample_rate, resampleQuality=resample_quality)

    audio_paths, emb_paths = get_input_output_paths(
        dataset=dataset,
        model_name=model_name,
        skip_clean=skip_clean,
        skip_deformed=skip_deformed,
        no_overwrite=no_overwrite,
        deform_list=deform_list,
    )

    for input_path, output_path in tqdm(
        zip(audio_paths, emb_paths), total=len(audio_paths)
    ):
        # Load audio
        monoloader.configure(filename=input_path)
        audio = monoloader()

        # Compute embeddings
        embedding = model(audio)

        if aggregation == "mean":
            embedding = np.mean(embedding, axis=0)
        elif aggregation is None:
            raise Exception(f"Aggregation method '{aggregation}' not implemented.")
        else:
            raise Exception(f"Aggregation method '{aggregation}' not implemented.")

        # Save embeddings
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, "wb") as f:
            np.save(f, embedding)


def generate_embeddings(
    dataset,
    model_name: str,
    skip_clean=False,
    skip_deformed=False,
    no_overwrite=False,
    deform_list=None,
    transcode_and_load=False,
):
    """Generate embeddings from a list of audio files.

    Args:
        dataset_cfg (dict): Dataset configuration.
        task_cfg (dict): Task configuration.
        aggregation (str, optional): Embedding aggregation method. Defaults to "mean".
        skip_clean (bool, optional): Whether to skip embedding generation for clean
                                     audio. Defaults to False.
        skip_deformed (bool, optional): Whether to skip embedding generation for
                                        deformed audio. Defaults to False.
        no_overwrite (bool, optional): Whether to skip embedding generation for
                                       existing embeddings. Defaults to False.
        deform_list (list, optional): List of deformation scenario indicies to include.
                                      If None, include all deformation scenarios.
    """

    aggregation = dataset.feature_aggregation
    # Load embedding model. Call essentia implementation if available,
    # otherwise custom implementation.

    if model_name == "vggish-audioset":
        model_path = "mir_ref/features/models/weights/audioset-vggish-3.pb"
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb",
                out="mir_ref/features/models/weights/",
            )
        check_model_exists(model_path)

        from essentia.standard import MonoLoader, TensorflowPredictVGGish

        model = TensorflowPredictVGGish(
            graphFilename=model_path, output="model/vggish/embeddings"
        )

        compute_and_save_embeddings(
            model=model,
            model_name=model_name,
            aggregation=aggregation,
            dataset=dataset,
            sample_rate=16000,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

    elif model_name == "effnet-discogs":
        model_path = (
            "mir_ref/features/models/weights/discogs_artist_embeddings-effnet-bs64-1.pb"
        )
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_artist_embeddings-effnet-bs64-1.pb",
                out="mir_ref/features/models/weights/",
            )
        check_model_exists(model_path)

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

        model = TensorflowPredictEffnetDiscogs(
            graphFilename=model_path, output="PartitionedCall:1"
        )

        compute_and_save_embeddings(
            model=model,
            model_name=model_name,
            aggregation=aggregation,
            dataset=dataset,
            sample_rate=16000,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

    elif model_name == "msd-musicnn":
        model_path = "mir_ref/features/models/weights/msd-musicnn-1.pb"

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb",
                out="mir_ref/features/models/weights/",
            )

        from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

        model = TensorflowPredictMusiCNN(
            graphFilename=model_path, output="model/dense/BiasAdd"
        )

        compute_and_save_embeddings(
            model=model,
            model_name=model_name,
            aggregation=aggregation,
            dataset=dataset,
            sample_rate=16000,
            resample_quality=4,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

    elif model_name == "maest":
        model_path = "mir_ref/features/models/weights/discogs-maest-30s-pw-1.pb"

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-1.pb",
                out="mir_ref/features/models/weights/",
            )

        check_model_exists(model_path)

        from essentia.standard import MonoLoader, TensorflowPredictMAEST

        model = TensorflowPredictMAEST(
            graphFilename=model_path, output="model/dense/BiasAdd"
        )

        compute_and_save_embeddings(
            model=model,
            model_name=model_name,
            aggregation=aggregation,
            dataset=dataset,
            sample_rate=16000,
            resample_quality=4,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

    elif model_name == "openl3":
        from essentia.standard import MonoLoader

        from mir_ref.features.models.openl3 import EmbeddingsOpenL3

        model_path = "mir_ref/features/models/weights/openl3-music-mel128-emb512-3.pb"

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.pb",
                out="mir_ref/features/models/weights/",
            )

        check_model_exists(model_path)

        extractor = EmbeddingsOpenL3(model_path)

        audio_paths, emb_paths = get_input_output_paths(
            dataset=dataset,
            model_name=model_name,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

        # Compute embeddings
        for input_path, output_path in tqdm(
            zip(audio_paths, emb_paths), total=len(audio_paths)
        ):
            embedding = extractor.compute(input_path)

            if aggregation == "mean":
                embedding = np.mean(embedding, axis=0)
            elif aggregation is None:
                pass
            else:
                raise Exception(f"Aggregation method '{aggregation}' not implemented.")

            # Save embeddings
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "wb") as f:
                np.save(f, embedding)

    elif model_name == "neuralfp":
        import tensorflow as tf
        from essentia.standard import MonoLoader

        mel_spec_model_dir = "mir_ref/features/models/weights/neuralfp/mel_spec"
        fp_model_dir = "mir_ref/features/models/weights/neuralfp/fp"
        mel_spec_model = tf.saved_model.load(mel_spec_model_dir)
        mel_spec_infer = mel_spec_model.signatures["serving_default"]
        fp_model = tf.saved_model.load(fp_model_dir)
        fp_infer = fp_model.signatures["serving_default"]

        monoloader = MonoLoader(sampleRate=8000, resampleQuality=1)

        audio_paths, emb_paths = get_input_output_paths(
            dataset=dataset,
            model_name=model_name,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

        for input_path, output_path in tqdm(
            zip(audio_paths, emb_paths), total=len(audio_paths)
        ):
            # Load audio
            monoloader.configure(filename=input_path)
            audio = monoloader()

            # Fingerprinting is done per 8000 samples, with a 4000 sample overlap, so pad
            audio = np.concatenate((np.zeros(4000), audio))
            audio = np.concatenate((audio, np.zeros(4000 - (len(audio) % 4000))))

            # Compute embeddings
            embeddings = []
            for buffer_start in range(0, len(audio) - 4000, 4000):
                buffer = audio[buffer_start : buffer_start + 8000]
                # size (None, 1, 8000)
                buffer.resize(1, 8000)
                buffer = np.array([buffer])
                # use mel spectrogram model
                mel_spec_emb = mel_spec_infer(tf.constant(buffer, dtype=tf.float32))[
                    "output_1"
                ]
                # use fingerprinter model
                fp_emb = fp_infer(mel_spec_emb)["output_1"]
                embeddings.append(fp_emb.numpy()[0])

            if aggregation == "mean":
                embedding = np.mean(embeddings, axis=0)
            elif aggregation is None:
                raise Exception(f"Aggregation method '{aggregation}' not implemented.")
            else:
                raise Exception(f"Aggregation method '{aggregation}' not implemented.")

            # Save embeddings
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "wb") as f:
                np.save(f, embedding)

    elif model_name == "mert-v1-330m" or model_name == "mert-v1-95m":
        # from transformers import Wav2Vec2Processor
        import torch
        import torchaudio.transforms as T
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        n_params = 330 if model_name == "mert-v1-330m" else 95
        # loading our model weights
        model = AutoModel.from_pretrained(
            f"m-a-p/MERT-v1-{n_params}M", trust_remote_code=True
        )
        # loading the corresponding preprocessor config
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"m-a-p/MERT-v1-{n_params}M", trust_remote_code=True
        )
        # get desired sample rate
        sample_rate = processor.sampling_rate

        audio_paths, emb_paths = get_input_output_paths(
            dataset=dataset,
            model_name=model_name,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

        if transcode_and_load:
            import librosa
            import sox

            tfm = sox.Transformer()
            tfm.convert(samplerate=sample_rate, n_channels=1)
        else:
            from essentia.standard import MonoLoader

            monoloader = MonoLoader(sampleRate=sample_rate, resampleQuality=1)

        for input_path, output_path in tqdm(
            zip(audio_paths, emb_paths), total=len(audio_paths)
        ):
            # Load audio
            if transcode_and_load:
                wav_input_path = input_path[:-4] + str(sample_rate) + ".wav"
                tfm.build(input_path, wav_input_path)
                audio, _ = librosa.load(wav_input_path, sr=sample_rate)
                os.remove(wav_input_path)
            else:
                monoloader.configure(filename=input_path)
                audio = monoloader()

            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            if aggregation == "mean":
                # we'll get the full embedding for now, meaning 13 layers x 768, or
                # 24 layers x 1024 for 95M and 330M respectively
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                embedding = all_layer_hidden_states.mean(-2).detach().cpu().numpy()
            elif aggregation is None:
                pass
            else:
                raise Exception(f"Aggregation method '{aggregation}' not implemented.")

            # Save embeddings
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "wb") as f:
                np.save(f, embedding)

    elif model_name == "clmr-v2":
        import subprocess

        import torch

        from mir_ref.features.models.clmr import SampleCNN, load_encoder_checkpoint

        # download model
        if not os.path.exists(
            "mir_ref/features/models/weights/clmr_checkpoint_10000.pt"
        ):
            print(f"Downloading {model_name} to mir_ref/features/models/weights...")
            wget.download(
                "https://github.com/Spijkervet/CLMR/releases/download/2.0/clmr_checkpoint_10000.zip",
                out="mir_ref/features/models/weights/",
            )

            # unzip clmr_checkpoint_10000
            subprocess.run(
                [
                    "unzip",
                    "mir_ref/features/models/weights/clmr_checkpoint_10000.zip",
                    "-d",
                    "mir_ref/features/models/weights/",
                ]
            )
            # delete zip
            subprocess.run(
                [
                    "rm",
                    "mir_ref/features/models/weights/clmr_checkpoint_10000.zip",
                ]
            )
            # delete clmr_checkpoint_10000_optim.pt
            subprocess.run(
                [
                    "rm",
                    "mir_ref/features/models/weights/clmr_checkpoint_10000_optim.pt",
                ]
            )

        # load model
        encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False)
        state_dict = load_encoder_checkpoint(
            "mir_ref/features/models/weights/clmr_checkpoint_10000.pt"
        )
        encoder.load_state_dict(state_dict)
        encoder.eval()

        audio_paths, emb_paths = get_input_output_paths(
            dataset=dataset,
            model_name=model_name,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

        if transcode_and_load:
            import librosa
            import sox

            tfm = sox.Transformer()
            tfm.convert(samplerate=22050, n_channels=1)
        else:
            from essentia.standard import MonoLoader

            monoloader = MonoLoader(sampleRate=22050, resampleQuality=1)

        for input_path, output_path in tqdm(
            zip(audio_paths, emb_paths), total=len(audio_paths)
        ):
            # Load audio
            if transcode_and_load:
                wav_input_path = input_path[:-4] + str(22050) + ".wav"
                tfm.build(input_path, wav_input_path)
                audio, _ = librosa.load(wav_input_path, sr=22050)
                os.remove(wav_input_path)
            else:
                monoloader.configure(filename=input_path)
                audio = monoloader()

            # get embedding per 59049 samples, padding the last buffer
            embeddings = []
            buffer_size = 59049
            for i in range(0, len(audio), buffer_size):
                buffer = audio[i : i + buffer_size]
                if len(buffer) < buffer_size:
                    buffer = np.pad(
                        buffer, (0, buffer_size - len(buffer)), mode="constant"
                    )
                buffer = torch.from_numpy(buffer).float()
                buffer = buffer.unsqueeze(0).unsqueeze(0)
                embedding = encoder(buffer).squeeze()
                embeddings.append(embedding)

            if aggregation == "mean":
                embedding = torch.mean(torch.stack(embeddings), axis=0).detach().numpy()
            elif aggregation is None:
                embedding = torch.stack(embeddings).detach().numpy()
            else:
                raise Exception(f"Aggregation method '{aggregation}' not implemented.")

            # Save embeddings
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "wb") as f:
                np.save(f, embedding)

    elif model_name == "mule":
        raise Exception("MULE embeddings are not fully supported yet.")

        from scooch import Config

        from mir_ref.features.models.mule import Analysis

        if aggregation is None:
            config = "mir_ref/features/models/weights/mule/mule_embedding_timeline.yml"
        elif aggregation == "mean":
            config = "mir_ref/features/models/weights/mule/mule_embedding_average.yml"
        else:
            raise Exception(f"Aggregation method '{aggregation}' not implemented.")

        cfg = Config(config)
        analysis = Analysis(cfg)

        audio_paths, emb_paths = get_input_output_paths(
            dataset=dataset,
            model_name=model_name,
            skip_clean=skip_clean,
            skip_deformed=skip_deformed,
            no_overwrite=no_overwrite,
            deform_list=deform_list,
        )

        for input_file, output_file in tqdm(
            zip(audio_paths, emb_paths), total=len(audio_paths)
        ):
            feat = analysis.analyze(input_file)
            feat.save(output_file)
