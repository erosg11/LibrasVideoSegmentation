import cv2
import numpy as np
import pytesseract.pytesseract as ocr
from imutils.object_detection import non_max_suppression
import re
import unidecode
from os.path import isfile, split
from pathlib import Path
from multiprocessing import Queue, Process, cpu_count
from tqdm import tqdm, trange
import pandas as pd
from numba import jit
from win32api import MessageBox
from VariationCalcMethods import CALC_VARIATIONS_METHODS

N_CPU = cpu_count()


class LibrasVideoSegmentation:
    SETUP = {"palavra", "alfabeto", "numerico", "frase", "naoembarcado"}

    def __init__(self, file: str, setup: str = 'alfabeto', out_folder='imagens', queue_size: int = 200):
        if not isfile(file):
            raise FileNotFoundError(f"File '{file}' not found")
        if setup not in self.SETUP:
            raise ValueError(f"Setup '{setup}' not exists")

        self.file = file
        self.setup = setup
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok=True)
        self.video = cv2.VideoCapture(file)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(round(self.video.get(cv2.CAP_PROP_FPS), 0))
        self.final_out_folder = self.out_folder / re.search(r'([^\\/]+)\.[^\\/]+$', self.file)[1]
        self.final_out_folder.mkdir(exist_ok=True)
        self.queue_size = queue_size
        self.cropping = False
        self.legend_location = None
        self.frame = None
        self.rect = None
        self.areas = []
        self.location_slice = None


    def calc_variations(self, in_queue, out_queue, calc_method: str = 'OTSU_NON_ZEROS', *, save=False):
        if calc_method not in CALC_VARIATIONS_METHODS:
            raise ValueError(f"""Method '{calc_method}' invalid, valid methods are '{"', '".join(
                CALC_VARIATIONS_METHODS.keys())}'""")

        self.video.set(1, 0)
        results_size = self.length - 2
        bins = [int(x) for x in np.linspace(0, results_size, N_CPU + 1, dtype='uint64')]

        for start, end in zip(bins[:-1], bins[1:]):
            in_queue.put((self.file, start, end, calc_method))

        results = np.zeros(results_size)
        try:
            for _ in trange(results_size, desc=f"Collecting results with method '{calc_method}'"):
                i, val = out_queue.get(timeout=3)
                results[i] = val
        except BaseException as e:
            print('Error', e, 'while collecting frames, probably missing frame')
        if save:
            np.save(str(self.final_out_folder / calc_method), results)
        return results

    def get_candidates(self, variations: np.ndarray, win_size=9):
        series = pd.Series(variations)
        window = series.rolling(win_size, center=True).sum()
        index_arr = np.asarray(window[(window >= 4.5) & (series > 1)].index.to_list())
        return np.append(self.find_start_frames(index_arr, self.fps), self.length)

    @staticmethod
    @jit(nopython=True)
    def find_start_frames(array: np.ndarray, distance: int):
        i = 0
        while i < array.size:
            val = array[i]
            if val:
                sliced_array = array[i + 1:]
                sliced_array[sliced_array < val + distance] = -1
            i += 1
        return array[array != -1]

    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.legend_location = [[x, y]]
            self.cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.legend_location.append([x, y])
            self.cropping = False
            refPt = self.legend_location
            # draw a rectangle around the region of interest
            if refPt[0][1] != refPt[1][1] and refPt[0][0] != refPt[1][0]:
                cv2.rectangle(self.frame, (refPt[0][0], refPt[0][1]), (refPt[1][0], refPt[1][1]), (0, 255, 0), 2)
                cv2.imshow("image", self.frame)

            for i in range(self.rect):
                if (refPt[0][1]) > self.areas[i][2] and (refPt[1][1]) < self.areas[i][3] and (refPt[0][0]) > \
                        self.areas[i][0] and (refPt[1][0]) < self.areas[i][1]:
                    cv2.rectangle(self.frame, (self.areas[i][0], self.areas[i][2]),
                                  (self.areas[i][1], self.areas[i][3]), (0, 255, 255), -1)
                    refPt[0][0] = self.areas[i][0] + 2  # descontando as bordas retângulo que foi inserido (espessura 2)
                    refPt[0][1] = self.areas[i][2] + 2
                    refPt[1][0] = self.areas[i][1] - 2
                    refPt[1][1] = self.areas[i][3] - 2

    @staticmethod
    def text_detection(image):
        orig = image.copy()
        (H, W) = image.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        # print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        retangulos = 0
        areas = []
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            areas.append([startX, endX, startY, endY])
            retangulos = retangulos + 1

        # show the output image
        # cv2.imshow("Text Detection", orig)
        # cv2.waitKey(0)
        return areas, retangulos, orig

    def find_coord(self, n_frame: int):
        position = n_frame + self.fps / 2
        self.video.set(1, position)
        _, self.frame = self.video.read()
        self.areas, self.rect, self.frame = self.text_detection(self.frame)
        MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
                      '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
                   'Identifique a área da legenda')
        clone = self.frame.copy()
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("image", lambda *args, **kwargs: self.click_and_crop(*args, **kwargs))
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.frame = clone.copy()
            if key == ord("n"):
                position += 1
                self.video.set(1, position)
                _, self.frame = self.video.read()
                clone = self.frame.copy()
            elif key == ord("c"):
                break
            elif key == 27:
                self.legend_location = []
                break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if self.legend_location and (self.legend_location[0][1] > self.legend_location[1][1]):
            self.legend_location[1][1], self.legend_location[0][1] = (self.legend_location[0][1],
                                                                      self.legend_location[1][1])

        if self.legend_location and (self.legend_location[0][0] > self.legend_location[1][0]):
            self.legend_location[1][0], self.legend_location[0][0] = (self.legend_location[0][0],
                                                                      self.legend_location[1][0])

        if self.legend_location and (self.legend_location[0][1] == self.legend_location[1][1] or
                                     self.legend_location[0][0] == self.legend_location[1][0]):
            print("Não é possível gerar um recorte a partir da área selecionada")

        self.legend_location = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]

        cv2.destroyAllWindows()
        self.location_slice = slice(*self.legend_location[2:]), slice(*self.legend_location[:2])
        return self.legend_location

    def save_frame(self, subtitle, candidate):
        pass

    def alphabet_processing(self, candidates):
        i = 0
        avanco = 0
        len_candidates = len(candidates)
        end = len_candidates - 1
        while i < end:
            self.video.set(1, candidates[i] + self.fps / 2 + avanco)
            frame = self.video.read()[1]
            slice_ = frame[self.location_slice]
            cv2.imshow("frame", slice_)
            cv2.waitKey(100)
            frame = cv2.cvtColor(slice_, cv2.COLOR_BGR2GRAY)
            frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            legenda = ocr.image_to_string(frame, lang="por", config='--psm 10')  # config psm 10 é para orientar a busca
            # por caracteres únicos
            legenda = re.sub(u'[^A-Z]', '', legenda.upper())

            len_legenda = len(legenda)
            if len_legenda == 1 and i < len_candidates:
                pass


if __name__ == '__main__':
    l = LibrasVideoSegmentation('E:\\Backups\\Cefet\\OneDrive - cefet-rj.br\\Dataset Lournco\\'
                                'Vídeos\\Alfabeto Editado\\Alfabeto56.mp4', queue_size=200)
    print(l.fps)
    # for method in CALC_VARIATIONS_METHODS:
        # variations = l.calc_variations(method, save=True)
    # variations = np.load('Variation_alfa56.npy')
    # print(variations)
    # np.save('Variation_alfa56', variations)
    # candidates = l.get_candidates(variations, 9)
    # print(list(candidates))
    # print(l.find_coord(candidates[0]))
