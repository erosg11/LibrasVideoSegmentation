import numpy as np
import cv2
from multiprocessing import Queue


__ALL__ = ['CALC_VARIATIONS_METHODS']


def _iter_frames(video: str, start: int, end: int):
    cap = cv2.VideoCapture(video)
    try:
        cap.set(1, start)
        for i in range(start - 1, end):
            ret, frame = cap.read()
            if not ret:
                print(f"\n\n\nError in frame {i} while loading file {video}.\n\n\n")
                continue
            yield i, frame
    finally:
        cap.release()


def _processes_equalize_hist(frame: np.ndarray):
    return cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def _calc_equalized_non_zero(video: str, start: int, end: int, out_queue: Queue):
    iter_frame = _iter_frames(video, start, end)
    last_frame = _processes_equalize_hist(next(iter_frame)[1])
    for i, new_frame in iter_frame:
        new_frame = _processes_equalize_hist(new_frame)
        res = cv2.absdiff(last_frame, new_frame).astype(np.uint8)
        out_queue.put((i, ((np.count_nonzero(res) * 100) / res.size) ** 2))
        last_frame = new_frame


def _calc_equalized_sum(video: str, start: int, end: int, out_queue: Queue):
    iter_frame = _iter_frames(video, start, end)
    last_frame = _processes_equalize_hist(next(iter_frame)[1])
    for i, new_frame in iter_frame:
        new_frame = _processes_equalize_hist(new_frame)
        res = cv2.absdiff(last_frame, new_frame).astype(np.uint8)
        out_queue.put((i, res.sum()))
        last_frame = new_frame


def _calc_square_equalized_sum(video: str, start: int, end: int, out_queue: Queue):
    iter_frame = _iter_frames(video, start, end)
    last_frame = _processes_equalize_hist(next(iter_frame)[1])
    for i, new_frame in iter_frame:
        new_frame = _processes_equalize_hist(new_frame)
        res = cv2.absdiff(last_frame, new_frame).astype(np.uint8)
        out_queue.put((i, res.sum()**2))
        last_frame = new_frame


def _process_otsu(frame: np.ndarray):
    return cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV |
                         cv2.THRESH_OTSU)[1]


def _calc_otsu_non_zero(video: str, start: int, end: int, out_queue: Queue):
    iter_frame = _iter_frames(video, start, end)
    last_frame = _process_otsu(next(iter_frame)[1])
    for i, new_frame in iter_frame:
        new_frame = _process_otsu(new_frame)
        res = cv2.absdiff(last_frame, new_frame).astype(np.uint8)
        out_queue.put((i, (np.count_nonzero(res) * 100) / res.size))
        last_frame = new_frame


def runner(in_queue: Queue, out_queue: Queue):
    while True:
        video, start, end, method = in_queue.get()
        CALC_VARIATIONS_METHODS[method](video, start, end, out_queue)


CALC_VARIATIONS_METHODS = {
    'OTSU_NON_ZEROS': _calc_otsu_non_zero,
    'EQUALIZED_NON_ZEROS': _calc_equalized_non_zero,
    'EQUALIZED_SUM': _calc_equalized_sum,
    'SQUARE_EQUALIZED_SUM': _calc_square_equalized_sum,
}

