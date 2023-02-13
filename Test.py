import itertools as it
import os
from timeit import timeit

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from Audio import Audio
from Model import Model
from Profile import Profile


def load(file_path: str, sample_rate: int, duration: int = 2):
    waveform = audio.load(file_path)
    length = sample_rate * duration
    waveforms = []
    for i in range(len(waveform) // length):
        waveforms.append(waveform[i * length:(i + 1) * length])

    return np.stack(waveforms, axis=0)


def infer_origin(model: Model, waveforms: np.array, sample_rate: int):
    embeddings = []
    for waveform in waveforms:
        embeddings.append(
            model.infer_one(waveform=waveform, sample_rate=sample_rate))

    return np.stack(embeddings, axis=0)


def infer(model: Model, waveforms: np.array):

    return model.infer(waveforms)


def profile_test(model: Model,
                 waveforms: np.array,
                 profile: Profile,
                 mode: int,
                 username: str = None):
    """
    warning: remember to enroll first, 
    after the test, delete the enrolled user
    mode: 0 for recognize, 1 for enroll, 2 for delete, 
    """
    assert mode in [0, 1, 2]
    assert 0 == mode or username is not None

    embeddings = infer(model, waveforms)
    if 0 == mode:
        user_score_sorted = profile.recognize(
            embedding=np.mean(embeddings, axis=0))
        if not user_score_sorted:
            print("no enrolled user")
        else:
            max_score = user_score_sorted[0][1]
            max_username = user_score_sorted[0][0]
            print(max_score, max_username)
    elif 1 == mode:
        print(profile.enroll(embeddings, username))
    elif 2 == mode:
        profile.delete(username)


if "__main__" == __name__:
    sample_rate = 16000
    modelname = "ECAPA_TDNN_GLOB_c512-ASTP-emb192-ArcMargin-LM"
    for config in modelname.split('-'):
        if "emb" in config:
            embeding_size = int(config[3:])

    audio = Audio(None, 3)
    model = Model(None, os.path.join("Model", f"{modelname}.onnx"), sample_rate)
    profile = Profile(None, os.path.join("Profile", f"{modelname}"),
                      embeding_size)

    dtype = model.dtype
    num_mel_bins = model.num_mel_bins
    frame_length = model.frame_length
    frame_shift = model.frame_shift
    dither = model.dither
    high_freq = model.high_freq
    low_freq = model.low_freq

    def fbank_origin(waveforms: torch.tensor):
        feats = []
        for waveform in waveforms:
            feats.append(
                kaldi.fbank(waveform.unsqueeze(0),
                            num_mel_bins=num_mel_bins,
                            frame_length=frame_length,
                            frame_shift=frame_shift,
                            dither=dither,
                            sample_frequency=sample_rate,
                            window_type='hamming',
                            use_energy=False))

        return torch.stack(feats, dim=0)

    def fbank(model: Model, waveforms: torch.tensor):

        return model.fbank(waveforms,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           dither=dither,
                           sample_frequency=sample_rate,
                           window_type='hamming',
                           use_energy=False)

    def plot(signals: np.ndarray, type: str, titles: list = None):
        assert type in ["waveshow", "linear", "fft", "hz", "log", "mel"]

        if titles is not None:
            n_row = len(titles)
        else:
            n_row = signals.shape[0]
            titles = []
            for i in range(n_row):
                titles.append(f"{i+1}-th signal")

        fig, axs = plt.subplots(nrows=n_row, sharex=True)
        max_value = np.finfo(signals.dtype).min
        hop_length = int(frame_shift * sample_rate // 1000)
        win_length = int(frame_length * sample_rate // 1000)
        global high_freq
        if high_freq <= 0.0:
            high_freq += 0.5 * sample_rate

        def helper(signal: np.ndarray, ax, title: str):
            if "waveshow" == type:
                img = librosa.display.waveshow(y=signal,
                                               sr=sample_rate,
                                               x_axis="ms",
                                               ax=ax)
            else:
                img = librosa.display.specshow(
                    data=signal,
                    x_axis="ms",
                    y_axis=type,
                    sr=sample_rate,
                    hop_length=hop_length,
                    n_fft=kaldi._next_power_of_2(win_length),
                    win_length=win_length,
                    fmin=low_freq,
                    fmax=high_freq,
                    ax=ax)

            ax.set(title=title)
            ax.label_outer()

            return img

        if 1 == n_row:
            img = helper(signals[0], axs, titles[0])
        else:
            for i in range(n_row):
                temp = helper(signals[i], axs[i], titles[i])
                value = np.max(signals[i])
                if value > max_value:
                    max_value = value
                    img = temp

        if "waveshow" != type:
            fig.colorbar(img, ax=axs, format="%+2.f")

        plt.show()

    # for recognize, enroll, delete testing
    # id_dir = r"D:\Graduate\Voice\Model\data\VoxCeleb1_wav\test"
    # filename = "00003"
    # for id in os.listdir(id_dir):
    #     for video in os.listdir(os.path.join(id_dir, id)):
    #         try:
    #             file_path = os.path.join(id_dir, id, video, f"{filename}.wav")
    #             waveforms = load(file_path, sample_rate)
    #         except:
    #             continue

    #         profile_test()

    #         break

    #         for wav in os.listdir(os.path.join(id_dir, id, video)):
    #             print(wav)

    # for filename in os.listdir("Audio"):
    #     try:
    #         file_path = os.path.join("Audio", f"{filename}")
    #         waveforms = load(file_path, sample_rate)
    #     except:
    #         continue

    #     profile_test(model, waveforms, profile, 2, filename.split('.')[0])

    waveforms = load(r"hzf_enroll.wav", sample_rate)

    # for fbank time testing
    # waveforms = torch.from_numpy(waveforms).to(dtype)
    # number = 100
    # print(timeit(stmt=lambda: fbank_origin(waveforms), number=number) / number)
    # print(timeit(stmt=lambda: fbank(model, waveforms), number=number) / number)

    # for fbank testing
    waveforms = torch.from_numpy(waveforms).to(dtype)
    feats = fbank_origin(waveforms)
    # feats = fbank(model, waveforms)
    plot(feats.numpy().reshape(feats.shape[0], feats.shape[2], feats.shape[1]),
         "mel", ["LFBE"])

    # for infer time testing
    # number = 10
    # print(
    #     timeit(stmt=lambda: infer_origin(model, waveforms, sample_rate),
    #            number=number) / number)
    # print(timeit(stmt=lambda: infer(model, waveforms), number=number) / number)

    # for infer testing
    # embeddings = infer_origin(model, waveforms, sample_rate)
    # embeddings = infer(model, waveforms)
    # string = ''
    # for i in range(embeddings.shape[0]):
    #     string += str(i)
    # for i in it.combinations(string, 2):
    #     x = embeddings[int(i[0])]
    #     y = embeddings[int(i[1])]
    #     print(model.cos_similarity(x, y))

    # for record, save testing
    # Audio.get_device_info()
    # audio.record(4)
    # audio.save("hzf.wav")
    audio.p.terminate()

    print("Done")
