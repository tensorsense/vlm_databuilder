import json
from typing import Optional,  TypeVar, Generic

import numpy as np
from langchain.pydantic_v1 import BaseModel, Field, PrivateAttr

from .time_utils import seconds_to_ts

# bound here doesn't work for some reason, but might have been fixed in pydantic v2
# https://github.com/pydantic/pydantic/issues/7774
OutputSchema = TypeVar('OutputSchema')#, bound=BaseModel)

class LLMInput(BaseModel, Generic[OutputSchema]):
    human_prompt: Optional[str|list[str]] = None
    system_prompt: Optional[str] = None
    _imgs: Optional[list[np.array]] = PrivateAttr()
    ntiles: Optional[int] = 1
    output_schema: Optional[OutputSchema] = None

    def __init__(self, _imgs=None, **data):
        self._imgs = _imgs
        super().__init__(**data)

class Segment(BaseModel, Generic[OutputSchema]):
    start_timestamp: str
    end_timestamp: str
    fps: float
    segment_info: Optional[OutputSchema]
    video_id: str
    _frames: Optional[list[np.array]] # List of raw frames that got into LLM. Added for debugging purposes.

    @classmethod
    def from_frames(cls, start_frame, end_frame, fps, **kwargs):
        return cls(start_timestamp=seconds_to_ts(start_frame/fps), end_timestamp=seconds_to_ts(end_frame/fps), fps=fps, **kwargs)

    @classmethod
    def from_seconds(cls, start_seconds, end_seconds, **kwargs):
        return cls(start_timestamp=seconds_to_ts(start_seconds), end_timestamp=seconds_to_ts(end_seconds), **kwargs)

    def to_str(self, skip: list[str] = []):
        # skip -> fields from segment_info
        # dict() works both with pydantic model and with with unparsed dict
        if self.segment_info:
            d = dict(self.segment_info)
            for s in skip:
                del d[s]
            d = ': ' + json.dumps(d)
        else:
            d = ''
        return f'{self.start_timestamp}-{self.end_timestamp}{d}'

def get_video_annotation_class(segment_annotation_schema: type[BaseModel]):
    class SegmentInfo(BaseModel):
        '''
        Annotation for a video segment.
        '''
        start_timestamp: str = Field(description='start timestamp of the segment in format HH:MM:SS.MS')
        end_timestamp: str = Field(description='start timestamp of the segment in format HH:MM:SS.MS')
        segment_annotation: segment_annotation_schema = Field(description='list of annotations for the segment')

    class VideoAnnotation(BaseModel):
        '''
        Segments of a video.
        '''
        segments: list[SegmentInfo] = Field(description='information about each segment')

    return VideoAnnotation