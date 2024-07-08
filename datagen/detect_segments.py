from typing import Optional, Callable
from tqdm import tqdm
# from pathlib import Path
from scenedetect import detect, ContentDetector, AdaptiveDetector # , HashDetector, ThresholdDetector, split_video_ffmpeg
from langchain_core.pydantic_v1 import BaseModel, Field

from .core.config import DatagenConfig
from .core.chat import ask
from .core.img_utils import get_frames
from .core.types import LLMInput, Segment
from .core.time_utils import seconds_to_ts

class SegmentInfo(BaseModel):
    '''Information about a segment'''
    filtering: bool = Field(description='Whether the person is doing any kind of squat exercise. The whole body of the person doing the squats must be shown in the image, not just some part like legs.')
    on_screen_text: Optional[str] = Field(description='Any kind of overlay text present in the images.', default=None)

def detect_segments(
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
        video_ids = [video_path.stem for video_path in config.get_videos()]
    
    video_ids_parsed = [x.stem for x in config.segment_dir.iterdir()]

    for video_id in tqdm(set(video_ids) - set(video_ids_parsed)):
        print(f'{video_id} - starting')
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
            frames_arr = get_frames(video_path.as_posix(), start.frame_num, end.frame_num, frames_per_segment)
            llm_input = LLMInput(system_prompt='You are given frames from a video and infromation on what to extract from them.', human_prompt=None, _imgs=frames_arr, ntiles=ntiles, output_schema=segment_info_schema)
            segment_info: Optional[BaseModel] = ask(llm_input=llm_input, config=config)
            if segment_info is None:
                print(f'Video {video_id} {seconds_to_ts(start.frame_num/start.framerate)}-{seconds_to_ts(end.frame_num/end.framerate)} segment not processed, skipping.')
                continue
            segment = Segment.from_frames(start_frame=start.frame_num, end_frame=end.frame_num, fps=start.framerate, segment_info=segment_info, video_id=video_id, _frames=frames_arr)
            
            # frames.append(frames_arr)
            segments.append(segment) # should frames be optional or not?
        config.save_segments(video_id, segments)
        print(f'{video_id} - done')
    
    # return segments, frames