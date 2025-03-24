import argparse
import os.path as osp
import re

from pathlib import Path
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

import ImageReward as RM
model = RM.load("ImageReward-v1.0")

class VideoPromptDataset(Dataset):

    def __init__(self, video_dir, prompt):

        # process video names
        video_dir = Path(video_dir)
        video_paths = list(video_dir.glob('*.mp4'))

        self.data = []
        for vid_path in video_paths:
            res = {
                'prompt': prompt,
                'video_path': vid_path,
            }
            self.data.append(res)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    return frames


def evaluate(args):
    # get video prompt pairs
    val_dataset = VideoPromptDataset(args.output_dir, args.prompt)

    # evaluate videos
    rewards_list = []

    # traverse over target videos
    for data in val_dataset:
        vid_frames = read_video(data['video_path']) # composed of 16 frames
        rewards_list.append(model.score(data['prompt'], vid_frames))

    rewards = torch.Tensor(rewards_list).mean()

    print("ImageRewards: {:.4f}".format(rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A cheetah is running across the savannah')
    parser.add_argument('--output_dir', type=str, default='outputs/inference/cheetah')

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    evaluate(args)
