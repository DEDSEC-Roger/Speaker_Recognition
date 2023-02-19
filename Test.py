import itertools as it
import os
import platform
from timeit import timeit

if "armv7l" in platform.platform().split('-'):
    print("can not perform plotting in this platform")
else:
    import librosa.display
    import matplotlib.pyplot as plt

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from Audio import Audio
from Model import Model
from Profile import Profile


def load(file_path: str, sample_rate: int, duration: int = 2):
    """
    file_path: str
    sample_rate: int
    duration: the duration for each np.array

    return: np.array with shape [bs, sample_rate * duration]
    """
    waveform = audio.load(file_path)
    length = sample_rate * duration
    waveforms = []
    for i in range(len(waveform) // length):
        waveforms.append(waveform[i * length:(i + 1) * length])

    return np.stack(waveforms, axis=0)


def infer_origin(model: Model, waveforms: np.array, sample_rate: int):
    """
    model: Model
    waveforms: np.array with shape [bs, samples]
    sample_rate: int

    return: embeddings with shape [bs, embedding_size]
    """
    embeddings = []
    for waveform in waveforms:
        embeddings.append(
            model.infer_one(waveform=waveform, sample_rate=sample_rate))

    return np.stack(embeddings, axis=0)


def infer(model: Model, waveforms: np.array):
    """
    model: Model
    waveforms: np.array with shape [bs, samples]

    return: embeddings with shape [bs, embedding_size]
    """

    return model.infer(waveforms)


def profile_test(embeddings: np.array,
                 profile: Profile,
                 mode: int,
                 username: str = None):
    """
    warning: remember to enroll first, after the test, delete the enrolled user
    embeddings: np.array with shape [bs, embedding_size]
    profile: Profile
    mode: int
    username: str
    mode: 0 for recognize, 1 for enroll, 2 for delete, 
    """
    assert mode in [0, 1, 2]
    assert 0 == mode or username is not None

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
    audio = Audio(None, 3)
    sample_rate = audio.sample_rate

    modelname = "ECAPA_TDNN_GLOB_c512-ASTP-emb192-ArcMargin-LM"
    for config in modelname.split('-'):
        if "emb" in config:
            embeding_size = int(config[3:])
    model = Model(None, os.path.join("Model", f"{modelname}.onnx"),
                  audio.sample_rate)
    profile = Profile(None, os.path.join("Profile", f"{modelname}"),
                      embeding_size)

    dtype = model.dtype
    num_mel_bins = model.num_mel_bins
    frame_length = model.frame_length
    frame_shift = model.frame_shift
    dither = model.dither
    window_type = model.window_type
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
                            window_type=window_type,
                            use_energy=False))

        return torch.stack(feats, dim=0)

    def fbank(model: Model, waveforms: torch.tensor, spectrogram: bool = False):

        return model.fbank(waveforms,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           dither=dither,
                           sample_frequency=sample_rate,
                           window_type=window_type,
                           use_energy=False,
                           spectrogram=spectrogram)

    def plot(signals: np.ndarray,
             type: str,
             titles: list = None,
             diff_scale=False):
        """
        signals: np.ndarray with shape [bs, F, T]
        type: "waveshow", "linear", "fft", "hz", "log", "mel", chose one
        titles: determine the number of display
        diff_scale: if True, plot colorbar for each display
        """
        assert type in ["waveshow", "linear", "fft", "hz", "log", "mel"]

        if isinstance(titles, str):
            titles = [titles]

        if titles is not None:
            n_row = len(titles)
        else:
            n_row = signals.shape[0]
            titles = []
            for i in range(n_row):
                titles.append(f"{i+1}-th signal")

        fig, axs = plt.subplots(nrows=n_row, sharex=True)
        hop_length = int(frame_shift * sample_rate // 1000)
        win_length = int(frame_length * sample_rate // 1000)
        global high_freq
        if high_freq <= 0.0:
            high_freq += 0.5 * sample_rate

        def helper(signal: np.ndarray, ax, title: str):
            if "waveshow" == type:
                signal = signal.astype(np.float32)
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
            max_value = np.min(signals)
            for i in range(n_row):
                temp = helper(signals[i], axs[i], titles[i])
                value = np.max(signals[i])
                if diff_scale:
                    fig.colorbar(temp, ax=axs[i], format="%+2.f")
                elif value > max_value:
                    max_value = value
                    img = temp

        if not diff_scale and "waveshow" != type:
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

    #         embeddings = infer(model, waveforms)
    #         profile_test(embeddings, profile, 2, filename.split('.')[0])

    #         break

    #         for wav in os.listdir(os.path.join(id_dir, id, video)):
    #             print(wav)

    # for filename in os.listdir("Audio"):
    #     try:
    #         file_path = os.path.join("Audio", f"{filename}")
    #         waveforms = load(file_path, sample_rate)
    #     except:
    #         continue

    #     embeddings = infer(model, waveforms)
    #     profile_test(embeddings, profile, 2, filename.split('.')[0])

    waveforms = load(r"hzf_enroll.wav", sample_rate)
    # plot(np.expand_dims(waveforms[0], axis=0), "waveshow", titles="waveform")

    # for spectrogram testing
    # waveforms = torch.from_numpy(waveforms).to(dtype)
    # specs = fbank(model, waveforms[0].unsqueeze(0), spectrogram=True)
    # specs = specs.numpy().reshape(specs.shape[0], specs.shape[2],
    #                               specs.shape[1])
    # spec_org = kaldi.spectrogram(waveforms,
    #                              raw_energy=False,
    #                              window_type=window_type)
    # spec_org = spec_org.numpy().reshape(spec_org.shape[1], spec_org.shape[0])

    # spec_list = []
    # spec_list.append(specs[0])
    # spec_list.append(librosa.amplitude_to_db(specs[0]))
    # spec_list.append(spec_org)
    # spec_list = np.stack(spec_list, axis=0)
    # plot(spec_list,
    #      "hz", ["Spectrogram", "Spectrogram_dB", "log power Spectrogram"],
    #      diff_scale=True)

    # for mel fbank testing
    freq = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
    melfb = librosa.filters.mel(sr=sample_rate,
                                n_fft=512,
                                n_mels=model.num_mel_bins,
                                fmin=20,
                                norm=None)
    plt.plot(freq, np.transpose(melfb))
    plt.title("Mel-filterbank")
    plt.show()

    # for fbank time testing
    # waveforms = torch.from_numpy(waveforms).to(dtype)
    # number = 100
    # print(timeit(stmt=lambda: fbank_origin(waveforms), number=number) / number)
    # print(timeit(stmt=lambda: fbank(model, waveforms), number=number) / number)

    # for fbank testing
    # waveforms = torch.from_numpy(waveforms).to(dtype)
    # feats = fbank_origin(waveforms)
    # feats = fbank(model, waveforms)
    # feats = feats.numpy().reshape(feats.shape[0], feats.shape[2],
    #                               feats.shape[1])
    # plot(feats, "mel")

    # for infer time testing
    # number = 5
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
