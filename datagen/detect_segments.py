from typing import Optional, Callable
from datetime import datetime

from tqdm import tqdm
# from pathlib import Path
from scenedetect import detect, ContentDetector, AdaptiveDetector # , HashDetector, ThresholdDetector, split_video_ffmpeg
from langchain_core.pydantic_v1 import BaseModel, Field
from moviepy.editor import VideoFileClip
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother

from .core.config import DatagenConfig
from .core.chat import ask
from .core.img_utils import get_frames
from .core.types import LLMInput, Segment
from .core.time_utils import seconds_to_ts

class SegmentInfo(BaseModel):
    '''Information about a segment'''
    filtering: bool = Field(description='Whether the person is doing any kind of squat exercise. The whole body of the person doing the squats must be shown in the image, not just some part like legs.')
    on_screen_text: Optional[str] = Field(description='Any kind of overlay text present in the images.', default=None)

def detect_segments_gpt(
        segment_info_schema: type[BaseModel],
        config: DatagenConfig,
        video_ids: Optional[list[str]] = None,
        detection_algorithm_factory: Optional[Callable[[], ContentDetector]] = None,
        frames_per_segment: int = 1,
        ntiles: int = 1,
        # return_frames = False,
        min_duration: Optional[float] = 1,
        max_duration: Optional[float] = 30,
    ):
    """Detect segments in a video and analyze them.

    Args:
        segment_info_schema (type[BaseModel]): A pydantic model with the fields to detect from the video.
        config (DatagenConfig): Config instance.
        video_ids (Optional[list[str]], optional): Which ids to process. If None, process all downloaded videos. Defaults to None.
        detection_algorithm_factory (Optional[Callable[[], ContentDetector]], optional): Which scenedetect algorithm to use. Reusing algorithm . If None, use AdaptiveDetector(). Defaults to None.
        frames_per_segment (int, optional): Number of frames to send to GPT. Currently max for GPT is 8 (need to double check). Defaults to 1.
        ntiles (int, optional): Number of 512x512 tiles per input image.. Defaults to 1.
        min_duration (Optional[float], optional): Minimal segment duration in seconds. Shorter segments will be discarded. Defaults to None (will not be used).
        max_duration (Optional[float], optional): Maximal segment duration in seconds. Longer segments will be discarded.. Defaults to None (will not be used).
    """

    if video_ids is None:
        video_ids = [video_path.stem for video_path in config.get_videos() if config.get_transcript_path(video_path.stem).exists()]
    
    video_ids_parsed = [x.stem for x in config.segment_dir.iterdir()]

    for video_id in tqdm(set(video_ids) - set(video_ids_parsed)):
        print(datetime.now(), f'{video_id} - starting')
        video_path = config.get_video_path(video_id)
        detection_algorithm_factory = detection_algorithm_factory or AdaptiveDetector
        detection_algorithm = detection_algorithm_factory()
        scene_list = detect(video_path.as_posix(), detection_algorithm, start_in_scene=True)

        segments: list[Segment] = []
        # frames = []
        for start, end in scene_list:
            duration = (end.frame_num - start.frame_num) / start.framerate
            if (min_duration and (min_duration > duration)) or (max_duration and (max_duration < duration)):
                continue
            try:
                frames_arr = get_frames(video_path.as_posix(), start.frame_num, end.frame_num, frames_per_segment)
            except:
                print(datetime.now(), f'Video {video_id} {seconds_to_ts(start.frame_num/start.framerate)}-{seconds_to_ts(end.frame_num/end.framerate)} video file error, skipping.')
                continue
            llm_input = LLMInput(system_prompt='You are given frames from a video and infromation on what to extract from them.', human_prompt=None, _imgs=frames_arr, ntiles=ntiles, output_schema=segment_info_schema)
            segment_info: Optional[BaseModel] = ask(llm_input=llm_input, config=config)
            if segment_info is None:
                print(datetime.now(), f'Video {video_id} {seconds_to_ts(start.frame_num/start.framerate)}-{seconds_to_ts(end.frame_num/end.framerate)} segment not processed, skipping.')
                continue
            segment = Segment.from_frames(start_frame=start.frame_num, end_frame=end.frame_num, fps=start.framerate, segment_info=segment_info, video_id=video_id, _frames=frames_arr)
            
            # frames.append(frames_arr)
            segments.append(segment) # should frames be optional or not?
        config.save_segments(video_id, segments)
        print(datetime.now(), f'{video_id} - done')
    
    # return segments, frames


def detect_segments_clip(
        config: DatagenConfig,
        model: AutoModel,
        processor: AutoProcessor,
        device: str = 'cuda',
        video_ids: Optional[list[str]] = None,
        fps_sampling: int = 1,
        text_prompts: str|list[str] = [],
        # return_frames = False,
        frames_per_batch: Optional[int] = None,
        min_duration: Optional[float] = 1,
        max_duration: Optional[float] = 30, # TODO: split or discard longer clips
        min_prob=0.1, # minimum clip probability to consider the match
        max_gap_seconds=1, # gaps of prob<min_prob that could be inside segment
        min_segment_seconds=3, # discard very short segments
        smooth_fraction=0.02, # smoothing strength
):
    if type(text_prompts) is str:
        text_prompts = [text_prompts]    

    if video_ids is None:
        video_ids = [video_path.stem for video_path in config.get_videos()]

    video_ids_parsed = [x.stem for x in config.segment_dir.iterdir()]
    
    for video_id in tqdm(set(video_ids) - set(video_ids_parsed)):
        video_path = config.get_video_path(video_id)

        video = VideoFileClip(video_path.as_posix())
        ticks = np.arange(1/fps_sampling/2, video.duration, 1/fps_sampling)
        print(datetime.now(), f'{video_id} - starting - {len(ticks)} frames')

        frames = [video.get_frame(s) for s in ticks]

        if frames_per_batch is None:
            frames_batched = [frames]
        else:
            frames_batched = [frames[i: i+frames_per_batch] for i in range(0, len(frames), frames_per_batch)]

        probs = []
        for i, frames_batch in enumerate(frames_batched):
            print(datetime.now(), f'{video_id} - batch {i} - starting')
            # TODO don't need to calculate text embedding for each batch, but it's a very minor optimization.
            inputs = processor(text=text_prompts, images=frames_batch, padding="max_length", return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs_batch = torch.sigmoid(logits_per_image).cpu().numpy()
            probs_batch = probs_batch.mean(axis=1)
            probs.append(probs_batch)
        print(datetime.now(), f'{video_id} - all batches done - detecting segments')
        probs = np.concatenate(probs)

        smoother = LowessSmoother(smooth_fraction=smooth_fraction, iterations=1)
        data = smoother.smooth(probs)
        segments_start_end = get_segments(data.smooth_data[0], max_gap=round(max_gap_seconds*fps_sampling), min_prob=min_prob, min_segment=round(min_segment_seconds*fps_sampling))
        
        segments = []
        for start, end in segments_start_end:
            segments.append(Segment.from_seconds(ticks[start], ticks[end], fps=video.fps, video_id=video_id))
        config.save_segments(video_id, segments)


def get_segments(data, max_gap=3, min_prob=0.1, min_segment=5):
    segments = []
    cur_segment_start = None
    not_doing = 0
    for i, p in enumerate(data):
        if p >= min_prob and cur_segment_start is None:
            cur_segment_start = i
        elif cur_segment_start is not None and p < min_prob:
            if not_doing >= max_gap:
                if i-not_doing - cur_segment_start >= min_segment:
                    segments.append((cur_segment_start, i-not_doing))
                not_doing = 0
                cur_segment_start = None
            else:
                not_doing += 1
        elif p >= min_prob:
            not_doing = 0
    if cur_segment_start is not None:
        segments.append((cur_segment_start, i-not_doing))

    return segments