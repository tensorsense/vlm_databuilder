import ffmpeg
from tqdm import tqdm
import json

from .core.config import DatagenConfig

def cut_videos(config: DatagenConfig, ann_file: str = 'annotations.json'):
    with open(config.data_dir / ann_file) as f:
        annotations = json.load(f)
    for ann in tqdm(annotations):
        output_filename = config.clip_dir / f"{ann['id']}.mp4"
        if output_filename.exists():
            print(output_filename.as_posix(), 'exists, skipping.')
            continue
        (ffmpeg
        .input(config.get_video_path(ann['video_id']), ss=ann['start_timestamp'], to=ann['end_timestamp'])
        .output(filename=output_filename, loglevel="quiet")
        .run())