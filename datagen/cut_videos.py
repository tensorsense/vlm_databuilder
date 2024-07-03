import ffmpeg
from tqdm import tqdm

from .core.config import DatagenConfig

def cut_videos(segments: list, config: DatagenConfig):
    for seg in tqdm(segments):
        # print(seg)
        output_filename = config.clip_dir / f"{seg['id']}.mp4"
        if output_filename.exists():
            print(output_filename.as_posix(), 'exists, skipping.')
            continue
        (
            ffmpeg
            .input(config.get_video_path(seg['video_id']), ss=seg['start_timestamp'], to=seg['end_timestamp'])
            .output(filename=output_filename, loglevel="quiet")
            .run()
        )