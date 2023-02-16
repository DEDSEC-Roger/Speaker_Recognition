import collections
import os
import platform
import sys
import typing
import wave

import numpy as np
import pyaudio
import PySide2
import webrtcvad
from PySide2.QtCore import QObject, Signal, Slot


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Audio(QObject):
    after_vad_signal = Signal(float)
    recorded_signal = Signal(type(np.ndarray))

    def __init__(self,
                 parent: typing.Optional[PySide2.QtCore.QObject] = ...,
                 mode: int = ...) -> None:
        """
        warning: run Audio.get_device_info() first, 
        to select the correct input_index and output_index
        mode: vad aggressiveness mode, [0, 3]
        """
        super().__init__(parent)

        self.sample_rate = 16000
        self.channels = 1
        self.width = 2
        if "armv7l" in platform.platform().split('-'):
            self.input_index = 2
            self.output_index = 2
        else:
            self.input_index = 1
            self.output_index = 3
        self.chunk = self.sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(rate=self.sample_rate,
                                  channels=self.channels,
                                  format=self.p.get_format_from_width(
                                      self.width),
                                  input=True,
                                  output=False,
                                  input_device_index=self.input_index,
                                  output_device_index=None,
                                  frames_per_buffer=0,
                                  start=False,
                                  stream_callback=None)
        self.min_record_duration = 3

        # for webrtcvad
        assert self.sample_rate in (8000, 16000, 32000, 48000)
        assert self.channels == 1
        assert self.width == 2
        assert mode in [0, 1, 2, 3]
        # The vad frame duration in milliseconds.
        self.frame_duration_ms = 30
        assert self.frame_duration_ms in (10, 20, 30)
        # The maxlen of deque to pad the window.
        self.maxlen = 10
        self.trigger_len = 10
        self.vad = webrtcvad.Vad(mode)

        self.running = True

    @Slot()
    def record(self, duration: int):
        """
        duration: seconds
        file_path: file path with extension

        return: np.array with dtype int16, shape [samples, ]
        """
        self.voiced_frame = bytes()
        voiced_duration = 0

        if duration < self.min_record_duration:
            self.record_duration = self.min_record_duration
            less_duration = self.min_record_duration
        else:
            self.record_duration = duration
            less_duration = int(duration / 3 * 2)

        num_chunk = int(self.sample_rate / self.chunk * self.record_duration)
        while self.running:
            self.stream.start_stream()
            for i in range(0, num_chunk):
                if not self.running:
                    self.stream.stop_stream()
                    return
                self.voiced_frame += self.stream.read(
                    self.chunk, exception_on_overflow=False)
            self.stream.stop_stream()

            self.voiced_frame = self.webrtcvad(self.voiced_frame)
            voiced_duration = len(self.voiced_frame) / (
                self.sample_rate * self.channels * self.width)

            if voiced_duration > less_duration:
                self.record_duration = less_duration

            if voiced_duration < duration:
                self.after_vad_signal.emit(duration - voiced_duration)
            else:
                break

        if not self.running:
            return

        # "<i2" indicates int16 with little endian
        voiced = np.frombuffer(self.voiced_frame, dtype="<i2")

        self.recorded_signal.emit(voiced)

        return voiced

    @Slot()
    def save(self, file_path: str):
        """
        file_path: file path with extension
        """
        file_count = 1
        file_dir, filename = os.path.split(file_path)
        filename, extension = filename.split('.')

        while os.path.exists(file_path):
            file_count += 1
            file_path = os.path.join(file_dir,
                                     f"{filename}_{file_count}.{extension}")

        with wave.open(file_path, 'wb') as wf:
            wf.setframerate(self.sample_rate)
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.width)
            wf.writeframes(self.voiced_frame)

    def load(self, file_path: str, play: bool = False):
        """
        file_path: file path with extension
        play: whether to play the audio

        return: np.array with dtype int16
        """
        with wave.open(file_path, 'rb') as wf:
            if play:
                stream = self.p.open(rate=wf.getframerate(),
                                     channels=wf.getnchannels(),
                                     format=self.p.get_format_from_width(
                                         wf.getsampwidth()),
                                     input=False,
                                     output=True,
                                     input_device_index=None,
                                     output_device_index=self.output_index,
                                     frames_per_buffer=0,
                                     start=True,
                                     stream_callback=None)

            frames = [wf.readframes(self.chunk)]
            while frames[-1]:
                frames.append(wf.readframes(self.chunk))
            frames = frames[:-1]

        if play:
            for i in range(len(frames)):
                stream.write(frames[i])
            stream.close()

        return np.frombuffer(b''.join(frames), dtype="<i2")

    def get_device_info():
        p = pyaudio.PyAudio()
        for i in range(0, p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels") > 0:
                print(f"input device: {info.get('index')} {info.get('name')}")
            if info.get("maxOutputChannels") > 0:
                print(f"output device: {info.get('index')} {info.get('name')}")

    # for webrtcvad, start from here
    def webrtcvad(self, audio: bytes):
        frames = self.frame_generator(audio)
        frames = list(frames)
        segments = self.vad_collector(frames)

        voiced_frame = bytes()
        for i, segment in enumerate(segments):
            if not self.running:
                return
            voiced_frame += segment

        return voiced_frame

    def frame_generator(self, audio: bytes):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(self.sample_rate * self.frame_duration_ms * self.width *
                self.channels // 1000)
        offset = 0
        timestamp = 0.0
        while offset + n <= len(audio):
            yield Frame(audio[offset:offset + n], timestamp,
                        self.frame_duration_ms)
            timestamp += self.frame_duration_ms
            offset += n

    def vad_collector(self, frames: list):
        """Filters out non-voiced audio frames.

        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.

        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.

        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.

        Arguments:
        frames - a source of audio frames (sequence or generator).

        Returns: A generator that yields PCM audio data.
        """
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=self.maxlen)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, self.sample_rate)

            # sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced >= self.trigger_len:
                    triggered = True
                    sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp, ))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced >= self.trigger_len:
                    sys.stdout.write('-(%s)' %
                                     (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])


if "__main__" == __name__:
    audio = Audio(None, 3)
    audio.p.terminate()

    print("Done")
