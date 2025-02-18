import argparse
from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append('./co-tracker')

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
from omegaconf import OmegaConf
from munkres import Munkres
from scipy.cluster.hierarchy import linkage, fcluster


def get_similarity_matrix(tracklets1, tracklets2):
    tracklets1 = tracklets1.flatten(1)
    tracklets2 = tracklets2.flatten(1)
    diffs = tracklets1.unsqueeze(0) - tracklets2.unsqueeze(1)

    dists = (diffs ** 2).mean(-1)
    return dists


def get_score(dists):
    # for each row find the nearest neighbor
    min_dist, _ = dists.min(dim=1)
    average_score = min_dist.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video_path, mask=None):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=50, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="evaluation/motion_fidelity_score_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/inference/cheetah")
    parser.add_argument("--source_video", type=str, default="test_data/car_turn/car_turn.mp4")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--model", type=str)

    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    model = CoTrackerPredictor(checkpoint=config.cotracker_model_path)
    model = model.cuda()

    video_dir = opt.output_dir
    video_paths = list(Path(video_dir).glob('*.gif'))
    if len(video_paths) == 0:
        video_paths = list(Path(video_dir).glob('*.mp4'))

    source_video = opt.source_video
    
    box_mask = None

    original_tracklets = get_tracklets(model, source_video, mask=box_mask)

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    scores = []

    print(video_dir, list(video_paths))

    for vid_path in video_paths:
        edit_tracklets = get_tracklets(model, vid_path, mask=box_mask)

        similarity_matrix = get_similarity_matrix(original_tracklets, edit_tracklets)
        similarity_scores_dict1 = get_score(similarity_matrix)
        similarity_scores_dict2 = get_score(similarity_matrix.t())

        score = similarity_scores_dict1["average_score"] + similarity_scores_dict2["average_score"]

        print("Video score: {:.8f}; Path: {}".format(score, vid_path))

        scores.append(score / height / width

motion_fidelity = sum(scores) / len(scores)

print("Motion fidelity score: ", motion_fidelity)
