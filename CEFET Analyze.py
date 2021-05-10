from LibrasVideoSegmentation import LibrasVideoSegmentation
from os.path import split
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from cupy import fft
import cupy as cp
# from cusignal import firwin
from scipy.signal import lfilter, firwin
import numpy as np
from json import dump
from multiprocessing import Queue, Process, cpu_count
from VariationCalcMethods import runner


BASE_PATH = r'E:\Backups\Cefet\OneDrive - cefet-rj.br\Dataset Lourenco\Vídeos\Base multilingue'
ENERGY_LIMIT = 0.75
RESULT_FOLDER = './results_cefet'
NUM_TAPS = 20


def read_labels(file='labels base cefet.csv'):
    return pd.read_csv(file)


if __name__ == '__main__':
    out_queue = Queue()
    in_queue = Queue()
    processes = [Process(target=runner, args=(in_queue, out_queue), daemon=True)
                 for _ in range(cpu_count())]
    for p in processes:
        p.start()
    df = read_labels()
    videos = df['video'].unique()
    base_path = Path(BASE_PATH)
    result_folder = Path(RESULT_FOLDER)
    result_folder.mkdir(parents=True, exist_ok=True)
    tq = tqdm(list(base_path.glob('*/*.mp4')), 'Videos: ')
    for video in tq:  # type: Path
        rel_video = video.relative_to(base_path)  # type: Path
        tq.write(f'Reading {rel_video}')
        video_folder = result_folder / rel_video.parent / rel_video.stem
        video_folder.mkdir(exist_ok=True, parents=True)
        file = video
        proc = LibrasVideoSegmentation(file.__str__())
        variation = proc.calc_variations(in_queue, out_queue, 'EQUALIZED_SUM')
        np.save(video_folder / 'variation', variation)
        variation = cp.asarray(variation)
        hz = 1 / proc.fps
        yfft_eq_sum = fft.fft(variation)
        xfft_eq_sum = fft.fftfreq(variation.size, hz)[:variation.size // 2]
        sum_energy_eq_sum = cp.cumsum(2.0 / variation.size * cp.abs(yfft_eq_sum[0:variation.size // 2]))
        eq_sum_limiar_pc_energy = sum_energy_eq_sum.max() * ENERGY_LIMIT
        frequency = float(xfft_eq_sum[sum_energy_eq_sum <= eq_sum_limiar_pc_energy].max())
        filter_coef = firwin(NUM_TAPS, frequency, fs=2/hz)
        result = lfilter(filter_coef, 1, cp.asnumpy(variation))
        np.save(video_folder / 'filtered', result)
        result = cp.asarray(result)
        subs = cp.diff(result)
        subs2 = cp.diff(subs)
        critical_points = (cp.sign(subs[1:]) != cp.sign(subs[:-1])).astype(bool)
        maxes = critical_points & (subs2 < 0).astype(bool)
        mins = critical_points & (subs2 > 0).astype(bool)
        cp.save(video_folder / 'maxes', maxes)
        cp.save(video_folder / 'mins', mins)
        cp.save(video_folder / 'diff', subs)
        cp.save(video_folder / 'diff2', subs2)
        with (video_folder / 'metadata.json').open('w') as fp:
            dump({
                'hz': hz,
                'frequency': frequency,
            }, fp)
