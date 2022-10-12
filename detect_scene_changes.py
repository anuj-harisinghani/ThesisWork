import os
import moviepy.editor
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.io.wavfile
import cv2



def frame_index_to_min_sec(frame_index):
    time = frame_index / frame_rate
    mm = time // 60
    ss = time % 60
    return int(mm), ss


# test_screen_video = '/media/anuj/UExtra/ScreenRec_6FU/Screen_EA-152_6FU.mp4'
# test_screen_video = '/media/anuj/UExtra/ScreenRec_6FU/Screen_EA-194_6FU.mp4'
# test_screen_video = '/media/anuj/UExtra/ScreenRec_6FU/Screen_HI-134_6FU.mp4'
test_screen_video = '/media/anuj/UExtra/ScreenRec_6FU/Screen_HI-220 6FU.mp4'
video = cv2.VideoCapture(test_screen_video)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
frame_rate = video.get(cv2.CAP_PROP_FPS)
duration_seconds = frame_count / frame_rate

thresholds = {'PupilCalib': 0.8, 'CookieTheft': 0.22, 'Reading': 0.13}
TASKS = ['PupilCalib', 'CookieTheft', 'Reading']
templates = {'PupilCalib': cv2.cvtColor(cv2.imread('/media/anuj/UExtra/pupil_cross_template.png'), cv2.COLOR_BGR2GRAY),
             'CookieTheft': cv2.cvtColor(cv2.imread('/media/anuj/UExtra/CT_template.png'), cv2.COLOR_BGR2GRAY),
             'Reading': cv2.cvtColor(cv2.imread('/media/anuj/UExtra/Reading_template.png'), cv2.COLOR_BGR2GRAY)}

FC = 0
VALS = []
timings = []
for task in TASKS:
    print('working on task', task)
    width, height = int(video.get(3)), int(video.get(4))
    last_frame = np.zeros((height, width))

    # means, totals = [], []
    start_frame_index = np.inf
    end_frame_index = 0
    task_fc = 0

    while FC < frame_count:
        ret, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_diff = abs(img - last_frame)
        last_frame = img

        # template matching
        template = templates[task]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        threshold = thresholds[task]
        flag = False
        if np.amax(res) > threshold:
            flag = True
            if FC < start_frame_index:
                start_frame_index = FC
            if FC > end_frame_index:
                end_frame_index = FC

        # print(fc, flag, min_val, max_val, start_frame_index, end_frame_index)
        print(FC, flag, task, min_val, max_val, start_frame_index, end_frame_index)
        VALS.append([FC, flag, task, min_val, max_val, start_frame_index, end_frame_index])

        if FC - end_frame_index > 200 and end_frame_index != 0:
            mm1, ss1 = frame_index_to_min_sec(start_frame_index)
            mm2, ss2 = frame_index_to_min_sec(end_frame_index)
            timings.append([task, '{}:{}'.format(mm1, ss1), '{}:{}'.format(mm2, ss2)])
            print('found timestamp for task {} - start {}:{} and end {}:{}'.format(task, mm1, ss1, mm2, ss2))
            break

        # cv2.namedWindow('input screen')
        # cv2.moveWindow('input screen', 800, 30)
        # cv2.imshow('input screen', frame_diff)

        # means.append(np.mean(frame_diff))
        # totals.append(np.sum(frame_diff))
        task_fc += 1
        FC += 1

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

    cv2.destroyAllWindows()
    # vals = np.array(vals)

VALS = np.array(VALS, dtype=object)