import argparse
import os.path as osp
import re

from pathlib import Path
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn import CosineSimilarity
from transformers import CLIPModel, AutoProcessor, AutoTokenizer

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


@torch.no_grad()
def encode_prompt(prompt, tokenizer, clip_model, device='cuda'):
    text_input = tokenizer(
        prompt,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt').to(device)
    text_features = clip_model.get_text_features(**text_input)[0]

    return text_features


@torch.no_grad()
def encode_frames(frames, clip_processor, clipmodel, device='cuda'):
    # Get the CLIP embeddings
    clip_inputs = clip_processor(images=frames, return_tensors="pt").to(device)
    clip_features = clipmodel.get_image_features(**clip_inputs)  # [bs, 512]

    return clip_features


def cal_metrics(
        text_feature,
        tgt_clip_features,
        cossim
    ):
    clip_t, t_cons = [], []

    num_tgt_frames = len(tgt_clip_features)

    # all scores are computed frame-wise
    for frame_idx in range(num_tgt_frames):
        clip_t.append(cossim(text_feature, tgt_clip_features[frame_idx]).item())
        
        if frame_idx > 0:
            t_cons.append(cossim(tgt_clip_features[frame_idx - 1], tgt_clip_features[frame_idx]).item())

    clip_t_score = sum(clip_t) / len(clip_t)
    t_cons_score = sum(t_cons) / len(t_cons)

    return clip_t_score, t_cons_score


def evaluate(args):
    device = args.device

    # load CLIP models
    tokenizer = AutoTokenizer.from_pretrained(args.clip_ckpt)
    clip_processor = AutoProcessor.from_pretrained(args.clip_ckpt)
    clipmodel = CLIPModel.from_pretrained(args.clip_ckpt)

    clipmodel.to(device)
    clipmodel.eval()

    # get video prompt pairs
    val_dataset = VideoPromptDataset(args.output_dir, args.prompt)

    # evaluate videos
    clip_t_list = []
    t_cons_list = []
    cossim = CosineSimilarity(dim=0, eps=1e-6)

    # traverse over target videos
    for data in val_dataset:
        text_feature = encode_prompt(data['prompt'], tokenizer, clipmodel, device=device)

        vid_frames = read_video(data['video_path']) # composed of 16 frames
        tgt_clip_features = encode_frames(vid_frames, clip_processor, clipmodel, device=device)

        clip_t_score, t_cons_score = \
            cal_metrics(
                text_feature=text_feature,
                tgt_clip_features=tgt_clip_features,
                cossim=cossim,
            )

        clip_t_list.append(clip_t_score)
        t_cons_list.append(t_cons_score)

    # average
    assert len(clip_t_list) > 0, "At least one frame is required for evaluation"
    clip_t = torch.Tensor(clip_t_list).mean()

    assert len(t_cons_list) > 0, "At least one pair of consecutive frames are required for evaluation"
    t_cons = torch.Tensor(t_cons_list).mean()

    # print the results
    print("CLIP-T: {clip_t:.4f}, T. Cons.: {t_cons:.4f}"\
          .format(clip_t=clip_t, t_cons=t_cons))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A cheetah is running across the savannah')
    parser.add_argument('--output_dir', type=str, default='outputs/inference/cheetah')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip_ckpt', type=str, default='openai/clip-vit-base-patch32')
    args = parser.parse_args()

    evaluate(args)
