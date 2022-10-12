import os
import moviepy.editor
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.io.wavfile

screen_path = '/media/anuj/UExtra/ScreenRec_6FU/'
video_path = '/media/anuj/UExtra/Video_6FU/'

screen_pids = [i.split(' ')[0].split('_')[1] for i in os.listdir(screen_path)]
video_pids = [i.split(' ')[0].split('_')[1] for i in os.listdir(video_path)]

inter_pids = set.intersection(set(screen_pids), set(video_pids))  # len(inter_pids) = 121  - (30, 91) patients healthy

screen_inter = []
video_inter = []
for pid in inter_pids:
    for path in os.listdir(screen_path):
        if pid in path:
            screen_inter.append(path)
    for path in os.listdir(video_path):
        if pid in path:
            video_inter.append(path)

PID_filenames = np.vstack((list(inter_pids), screen_inter, video_inter)).transpose()
PID_filenames_df = pd.DataFrame(PID_filenames, columns=['PID', 'screen_file', 'video_file'], index=None)
PID_filenames_df.to_csv('/media/anuj/UExtra/pid_filenames.csv', index=False)

screen_lens = {}
video_lens = {}

for pid in tqdm(inter_pids):
    for path in video_inter:
        if pid in path:
            file_path = os.path.join(video_path, path)
            video = moviepy.editor.VideoFileClip(file_path)
            video_lens[pid] = video.duration

    for path in screen_inter:
        if pid in path:
            file_path = os.path.join(screen_path, path)
            screen = moviepy.editor.VideoFileClip(file_path)
            screen_lens[pid] = screen.duration

diffs = {pid: round(screen_lens[pid] - video_lens[pid], 4) for pid in inter_pids}
diffs2 = pd.DataFrame(diffs.values(), columns=['screen_minus_video_len'], index=diffs.keys())
diffs2['absolute'] = abs(diffs2.screen_minus_video_len)

less_than_10_seconds_diff = diffs2[diffs2.absolute < 10]  # len = 97
sorted_less_than_10 = less_than_10_seconds_diff.sort_values('absolute')


def count_patient_healthy(df):
    p, h = 0, 0
    if type(df) == pd.DataFrame:
        df = list(df.index)
    for pid in df:
        if pid.startswith('E'):
            p += 1
        if pid.startswith('H'):
            h += 1
    return p, h


n_patients, n_healthy = count_patient_healthy(sorted_less_than_10)
# (21, 76) patients vs healthy  ... (30, 91) for inter_pids  .... lost - (9, 15)
'''
PIDs with fautly video files:
HL-111
HI-132

Sending Andrea these PIDs to timestamp:
EA-152
EA-194
HI-134
HI-220
'''

def create_run_file_for_syncstart(PID_filenames: np.array):
    """
    function to create a run file for using with syncstart, only for the PIDs in inter_pids
    That run file will be executed to run with syncstart and to find the offset required in seconds for all these pids
    """
    filepath = '/media/anuj/UExtra/exec_syncstart.sh'
    with open(filepath, 'a') as file:
        for i in range(len(PID_filenames)):
            cmd = 'syncstart -s ScreenRec_6FU/{} Video_6FU/{}\n'.format(PID_filenames[i][1], PID_filenames[i][2])
            file.write(cmd)
    file.close()


'''
once syncstart has done it's work, now I have a list of video files with the offset values to make them sync together
'''
offset_file = open('/media/anuj/UExtra/offset.txt', 'r')
offsets = offset_file.readlines()


'''
section for finding timestamps
Overall idea:
1. import screen recording video
2. use scikit-video to convert the video into a numpy array
3. use frame(i) - frame(i-1) to get a difference in frames
4. find the threshold in which a huge enough change in frames denotes the start and end of a task
5. mark all timestamps where these huge changes are found - check sampling rate of the video for this and keep track of time elapsed
'''

# videodata = skvideo.io.vreader(test_screen_video)
# metadata = skvideo.io.ffprobe(test_screen_video)
#
# duration_ts, duration = float(metadata['video']['@duration_ts']), float(metadata['video']['@duration'])
#
# height, width, bits = int(metadata['video']['@height']), int(metadata['video']['@width']), 3
# last_frame = np.zeros((height, width, bits))
#
# means, totals = [], []
# for frame in videodata:
#     frame_diff = abs(frame - last_frame)
#     means.append(np.mean(frame_diff))
#     totals.append(np.sum(frame_diff))
#     last_frame = frame
