import typing

import numpy as np
import onnxruntime as ort
import PySide2
import torch
import torch.fft
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from PySide2.QtCore import QObject, Signal, Slot


class Model(QObject):
    inferred_signal = Signal(type(np.ndarray))
    inferred_one_signal = Signal(type(np.ndarray))

    def __init__(self,
                 parent: typing.Optional[PySide2.QtCore.QObject] = ...,
                 file_path: str = ...,
                 sample_frequency: int = ...) -> None:
        """
        file_path: onnx file path
        """
        super().__init__(parent)

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(file_path, sess_options=so)

        self.blackman_coeff = 0.42
        self.dither = 0.0
        self.frame_length = 25.0
        self.frame_shift = 10.0
        self.high_freq = 0.0
        self.low_freq = 20.0
        self.num_mel_bins = 80
        self.preemphasis_coefficient = 0.97
        self.remove_dc_offset = True
        self.sample_frequency = sample_frequency
        self.use_energy = False
        self.use_log_fbank = True
        self.use_power = True
        self.vtln_high = -500.0
        self.vtln_low = 100.0
        self.vtln_warp = 1.0
        self.window_type = "hamming"

        self.device = "cpu"
        self.dtype = torch.float32
        self.window_size = int(self.sample_frequency * self.frame_length *
                               0.001)
        # size (window_size, )
        self.window_function = kaldi._feature_window_function(
            self.window_type, self.window_size, self.blackman_coeff,
            self.device, self.dtype)

        self.padded_window_size = kaldi._next_power_of_2(self.window_size)
        # size (num_mel_bins, padded_window_size // 2)
        self.mel_energies, _ = kaldi.get_mel_banks(
            self.num_mel_bins, self.padded_window_size, self.sample_frequency,
            self.low_freq, self.high_freq, self.vtln_low, self.vtln_high,
            self.vtln_warp)

    @Slot()
    def infer(self, waveform: np.ndarray):
        """
        waveform: np.ndarray of shape [bs, samples]
        
        return: embeddings, with shape [bs, embedding_size]
        """
        assert waveform.ndim == 2, f"waveform.ndim shoule be 2, got {waveform.ndim}"

        waveform = torch.from_numpy(waveform).to(self.dtype)

        feats = self.fbank(
            waveform,
            blackman_coeff=self.blackman_coeff,
            dither=self.dither,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            high_freq=self.high_freq,
            low_freq=self.low_freq,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sample_frequency,
            use_energy=self.use_energy,
            vtln_high=self.vtln_high,
            vtln_low=self.vtln_low,
            vtln_warp=self.vtln_warp,
            window_type=self.window_type,
        )
        # CMN, without CVN
        feats -= torch.mean(feats, dim=1, keepdim=True)
        feats = feats.numpy()

        embeddings = self.session.run(output_names=['embs'],
                                      input_feed={'feats': feats})[0]

        self.inferred_signal.emit(embeddings)

        return embeddings

    @Slot()
    def infer_one(self,
                  file_path: str = None,
                  waveform: np.array = None,
                  sample_rate: int = None,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
        """
        file_path: file path with extension
        waveform: np.ndarray of shape [samples, ]
        sample_rate: waveform's sample_rate
        
        return: embedding, with shape [embedding_size, ]
        """
        assert waveform.ndim == 1, f"waveform.ndim shoule be 1, got {waveform.ndim}"
        assert file_path is not None or (
            waveform is not None and sample_rate is not None
        ), "at lease give file_path or (waveform and sample_rate)"

        if file_path is not None:
            waveform, sample_rate = torchaudio.load(file_path, normalize=False)
        else:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(self.dtype)

        feats = kaldi.fbank(waveform,
                            num_mel_bins=num_mel_bins,
                            frame_length=frame_length,
                            frame_shift=frame_shift,
                            dither=dither,
                            sample_frequency=sample_rate,
                            window_type='hamming',
                            use_energy=False)
        # CMN, without CVN
        feats -= torch.mean(feats, dim=0, keepdim=True)
        # add batch dimension
        feats = feats.unsqueeze(0).numpy()

        embedding = self.session.run(output_names=['embs'],
                                     input_feed={'feats': feats})[0].squeeze(0)

        self.inferred_one_signal.emit(embedding)

        return embedding

    def cos_similarity(self, x: np.array, y: np.array):
        assert x.ndim == 1 and x.shape == y.shape

        x_norm = np.linalg.norm(x, ord=2)
        y_norm = np.linalg.norm(y, ord=2)

        return np.dot(x, y) / (x_norm * y_norm)

    # for kaldi.fank with batch_size, start from here
    def fbank(self,
              waveform: torch.Tensor,
              blackman_coeff: float = 0.42,
              channel: int = -1,
              dither: float = 0.0,
              energy_floor: float = 1.0,
              frame_length: float = 25.0,
              frame_shift: float = 10.0,
              high_freq: float = 0.0,
              htk_compat: bool = False,
              low_freq: float = 20.0,
              min_duration: float = 0.0,
              num_mel_bins: int = 23,
              preemphasis_coefficient: float = 0.97,
              raw_energy: bool = True,
              remove_dc_offset: bool = True,
              round_to_power_of_two: bool = True,
              sample_frequency: float = 16000.0,
              snip_edges: bool = True,
              subtract_mean: bool = False,
              use_energy: bool = False,
              use_log_fbank: bool = True,
              use_power: bool = True,
              vtln_high: float = -500.0,
              vtln_low: float = 100.0,
              vtln_warp: float = 1.0,
              window_type: str = "povey") -> torch.Tensor:
        """
        waveform: torch.Tensor of shape [bs, samples]

        return: features, with shape [bs, T, F]
        """
        device, dtype = waveform.device, waveform.dtype

        window_shift = int(sample_frequency * frame_shift * 0.001)
        window_size = self.window_size
        padded_window_size = self.padded_window_size

        # strided_input, size (bs, m, padded_window_size)
        strided_input = self._get_window(waveform, padded_window_size,
                                         window_size, window_shift, window_type,
                                         blackman_coeff, snip_edges, raw_energy,
                                         energy_floor, dither, remove_dc_offset,
                                         preemphasis_coefficient)

        # size (bs, m, padded_window_size // 2 + 1)
        try:
            spectrum = torch.fft.rfft(strided_input, dim=-1, norm=None).abs()
        except:
            strided_input = np.fft.rfft(strided_input.numpy(),
                                        axis=-1,
                                        norm=None)
            spectrum = torch.from_numpy(strided_input).abs().to(self.dtype)

        if use_power:
            spectrum = spectrum.pow(2.0)

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies = self.mel_energies
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension,
        # size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = torch.nn.functional.pad(mel_energies, (0, 1),
                                               mode='constant',
                                               value=0)

        # sum with mel fiterbanks over the power spectrum, size (bs, m, num_mel_bins)
        # mel_energie = []
        # for spec in spectrum:
        #     mel_energie.append(torch.mm(spec, mel_energies.T))
        # mel_energies = torch.stack(mel_energie, dim=0)
        mel_energies = torch.matmul(spectrum, mel_energies.T)

        if use_log_fbank:
            # avoid log of zero (which should be prevented anyway by dithering)
            mel_energies = torch.max(mel_energies,
                                     kaldi._get_epsilon(device, dtype)).log()

        return mel_energies

    def _get_window(self, waveform: torch.Tensor, padded_window_size: int,
                    window_size: int, window_shift: int, window_type: str,
                    blackman_coeff: float, snip_edges: bool, raw_energy: bool,
                    energy_floor: float, dither: float, remove_dc_offset: bool,
                    preemphasis_coefficient: float):
        r"""Gets a window and its log energy

        Returns:
            Tensor: strided_input of size (m, ``padded_window_size``)
        """
        num_samples = waveform.size(1)
        strides = (window_shift * waveform.stride(1), waveform.stride(1))
        if num_samples < window_size:
            return torch.empty((0, 0),
                               dtype=waveform.dtype,
                               device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
        sizes = (m, window_size)

        # strided_input = []
        # for wave in waveform:
        #     strided_input.append(torch.as_strided(wave, sizes, strides))
        # strided_input = torch.stack(strided_input, dim=0)
        strided_input = torch.as_strided(waveform,
                                         (waveform.size(0), sizes[0], sizes[1]),
                                         (num_samples, strides[0], strides[1]))

        if remove_dc_offset:
            # Subtract each row/frame by its mean
            row_means = torch.mean(strided_input, dim=2, keepdim=True)
            strided_input = strided_input - row_means

        if preemphasis_coefficient != 0.0:
            # size (m, window_size + 1)
            offset = torch.nn.functional.pad(strided_input, (1, 0),
                                             mode='replicate')
            # strided_input[i,j] -=
            # preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
            strided_input -= preemphasis_coefficient * offset[:, :, :-1]

        # Apply window_function to each row/frame
        # size (m, window_size)
        strided_input = strided_input * self.window_function.unsqueeze(
            0).unsqueeze(0)

        # Pad columns with zero until we reach size (m, padded_window_size)
        if padded_window_size != window_size:
            padding_right = padded_window_size - window_size
            strided_input = torch.nn.functional.pad(strided_input,
                                                    (0, padding_right),
                                                    mode='constant',
                                                    value=0)

        return strided_input


if "__main__" == __name__:
    model = Model(None, r"Model\ResNet34-ASTP-emb256-ArcMargin.onnx", 16000)

    print("Done")
