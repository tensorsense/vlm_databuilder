from typing import Optional, List
from pathlib import Path
from scenedetect import detect, ContentDetector, AdaptiveDetector # , HashDetector, ThresholdDetector, split_video_ffmpeg
from langchain_core.pydantic_v1 import BaseModel, Field

from .core.config import DatagenConfig
from .core.chat import ask
from .core.img_utils import get_frames
from .core.types import LLMInput, Segment

class SegmentInfo(BaseModel):
    '''Information about a segment'''
    filtering: bool = Field(description='Whether the person is doing any kind of squat exercise. The whole body of the person doing the squats must be shown in the image, not just some part like legs.')
    on_screen_text: Optional[str] = Field(description='Any kind of overlay text present in the images.', default=None)

def detect_segments(
        video_id: str,
        segment_info_schema: type[BaseModel],
        config: DatagenConfig,
        detection_algorithm: Optional[ContentDetector],
        frames_per_segment: int = 1,
        ntiles: int = 1,
        # return_frames = False,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
    """Detect segments in a video and analyze them.

    Args:
        video_name (str): video file name (abc.mp4)
        segment_info_schema (type[BaseModel]): A pydantic model with the fields to detect from the video.
        config (DatagenConfig): Config instance.
        detection_algorithm(scenedetect.ContentDetector): an instance of a detetion algorithm from scenedetect library. Will have significant effect on resulting video clips depending on the kind of content. Defaults to AdaptiveDetector() with default parameters.
        frames_per_segment (int, optional): Number of frames to send to GPT. Currently max for GPT is 8 (need to double check).
        ntiles (int, optional): Number of 512x512 tiles per input image. Defaults to 1.
        min_duration (float, optional): Minimal segment duration in seconds. Shorter segments will be discarded.
        max_duration (float, optional): Maximal segment duration in seconds. Longer segments will be discarded.
    """
    if config.get_segment_path(video_id).exists():
        print(f'Segment json for {video_id} exist, skipping')
        return

    video_path = config.get_video_path(video_id)
    if detection_algorithm is None:
        # reusing one detector instance leads to some errors
        detection_algorithm = AdaptiveDetector()
    scene_list = detect(video_path.as_posix(), detection_algorithm, start_in_scene=True)

    segments: List[Segment] = []
    frames = []
    for start, end in scene_list:
        duration = (end.frame_num - start.frame_num) / start.framerate
        if (min_duration and (min_duration > duration)) or (max_duration and (max_duration < duration)):
            continue
        frames_arr = get_frames(video_path.as_posix(), start.frame_num, end.frame_num, frames_per_segment)
        llm_input = LLMInput(system_prompt='You are given frames from a video and infromation on what to extract from them.', human_prompt=None, _imgs=frames_arr, ntiles=ntiles, output_schema=segment_info_schema)
        segment_info: BaseModel = ask(llm_input=llm_input, config=config)
        segment = Segment.from_frames(start_frame=start.frame_num, end_frame=end.frame_num, fps=start.framerate, segment_info=segment_info, video_id=video_id, _frames=frames_arr)
        frames.append(frames_arr)
        segments.append(segment) # should frames be optional or not?
    config.save_segments(video_id, segments)
    return segments, frames