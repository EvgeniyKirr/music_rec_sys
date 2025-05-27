import ffmpeg
import numpy as np
from io import BytesIO
from PIL.ImageOps import crop
from os import remove, listdir
from librosa import power_to_db
from torch import Tensor, stack
from torchaudio import load as wav_load
from matplotlib import use, pyplot as plt
from PIL.Image import Image, open as img_open, merge
from torchvision.transforms import Compose, ToTensor, Normalize
from torchaudio.transforms import Spectrogram, SpectralCentroid


use('Agg')

class WavLoader():
    def __init__(self, wav_path: str) -> None:
        self._load_waveform(wav_path)
        self._make_frames()

    def _load_waveform(self, wav_path: str) -> None:
        wav_tmp = 'wav_tmp.mp3'
        ffmpeg.input(wav_path).output(wav_tmp, ac=1, loglevel="quiet").run()
        wf_tensor, self.sample_rate = wav_load(wav_tmp)
        self.waveform = wf_tensor[0]
        remove(wav_tmp)

    def _draw(self, array: np.ndarray, fig_size: tuple, crop_size: tuple) -> Image:
        fig = plt.figure(figsize=fig_size, dpi=44)
        fig.add_axes((0, 0, 1, 1))

        match array.ndim:
            case 1: plt.plot(np.arange(len(array)), array, color='black')
            case 2: plt.imshow(array, cmap='gray', origin="lower", aspect="auto")

        buf = BytesIO()
        plt.savefig(buf)
        plt.close(fig)
        img = img_open(buf)
        img = crop(img, crop_size)

        return img.split()[0]

    def _draw_waveform(self, wf: Tensor) -> Image:
        array = wf.numpy()
        img = self._draw(array, (9, 8), (22, 0, 22, 0))

        return img

    def _draw_spec_cent(self, wf: Tensor) -> Image:
        transform_sc = SpectralCentroid(self.sample_rate)
        spec_cent = transform_sc(wf).numpy()
        array = np.convolve(spec_cent, np.ones(10)/10, mode='same')
        img = self._draw(array, (9, 8), (22, 0, 22, 0))

        return img

    def _draw_spectrogram(self, wf: Tensor) -> Image:
        transform_sg = Spectrogram()
        array = (1 - power_to_db(transform_sg(wf)))
        img = self._draw(array, (8, 8), (0, 0, 0, 0))

        return img

    def _make_frames(self, frame_duration: int = 6) -> None:
        self.frames = []
        window_size = self.sample_rate * frame_duration
        for inx in range(0, len(self.waveform), window_size):
            crop_wf = self.waveform[inx:inx+window_size]
            if len(crop_wf) != window_size: break

            frame = merge('RGB', (
                self._draw_waveform(crop_wf),
                self._draw_spec_cent(crop_wf),
                self._draw_spectrogram(crop_wf)))

            self.frames.append(frame)

    def get_tensor(self) -> Tensor:
        if len(self.frames) < 24:
            return Tensor([])

        array = []
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
        middle_frames = len(self.frames) // 2
        for inx in range(middle_frames-10, middle_frames+10):
            array.append(transform(self.frames[inx]))

        return stack(array)

    def save_frames(self, path: str, id: int, cnt: int = 0) -> None:
        if len(self.frames) < 24: return

        middle_frames = len(self.frames) // 2
        if cnt == 0:
            cnt = len(self.frames) - 4

        for inx in range(middle_frames-(cnt//2), middle_frames+(cnt//2)):
            self.frames[inx].save(f'{path}\\{id:03}{inx:02}.png')


class FramesLoader():
    def __init__(self, dir_path: str) -> None:
        self.frames = []
        for file in listdir(dir_path):
            if '.png' in file:
                self.frames.append(img_open(f'{dir_path}\\{file}'))

    def get_tensor(self) -> Tensor:
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
        return stack([transform(frame) for frame in self.frames])