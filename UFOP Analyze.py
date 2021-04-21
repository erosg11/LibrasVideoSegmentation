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


BASE_PATH = '/media/eros/Mercúrio/desenvolvimento/datasets/UFOP/LIBRAS-UFOP DATASET'
ENERGY_LIMIT = 0.75
RESULT_FOLDER = './results'
NUM_TAPS = 20


def read_labels(file='labels.csv'):
    return pd.read_csv(file)


if __name__ == '__main__':
    df = read_labels()
    videos = df['video'].unique()
    base_path = Path(BASE_PATH)
    result_folder = Path(RESULT_FOLDER)
    result_folder.mkdir(parents=True, exist_ok=True)
    tq = tqdm(list(base_path.glob('p*_c*_s*')), 'Videos: ')
    for folder in tq:
        base_dir = split(folder)[1]
        if base_dir not in videos:
            continue
        tq.write(f'Reading {base_dir}')
        video_folder = result_folder / base_dir
        video_folder.mkdir(exist_ok=True)
        file = folder / "Color.avi"
        proc = LibrasVideoSegmentation(file.__str__())
        variation = proc.calc_variations('EQUALIZED_SUM')
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
