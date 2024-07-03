import json

from langchain.pydantic_v1 import BaseModel

from .core.types import Segment, LLMInput
from .core.config import DatagenConfig
from .core.chat import ask
from .core.types import get_video_annotation_class

system_prompt_default = '''You are a video annotator.
You are given a full transcript of a video in format of "<HH.MM.SS>\\n<text>" and a list of relevant segments in format "<HH:MM:SS.ms>-<HH:MM:SS.ms>:<json_info>".
Your task is to find relevant segments (according to user instructions) based on text transcript and other provided information.
The only segments that you can output are those that the user has provided the timestamps for.
Do not output a segment if the information requested by the user is not present in the transcript or segment descriptions.
'''

def generate_annotations(segments: list[Segment], annotation_schema: type[BaseModel], config: DatagenConfig, system_prompt: str = system_prompt_default):
    # assert llm_input.output_schema, "output schema must be present"

    video_ids = set([s.video_id for s in segments])
    outputs = {}
    for video_id in video_ids:
        if config.get_anno_path(video_id).exists():
            print(f'Annotation {video_id} exists, skipping.')
            continue
        video_segments = [s for s in segments if s.video_id==video_id]
        
        prompt = []
        prompt.append('Segment information:\n' + '\n'.join([s.str_format for s in video_segments]))
        transcript = config.get_transcript(video_id)
        if transcript:
            prompt.append('Transcript:\n' + transcript)
        # print(VideoAnnotation[annotation_schema], type(VideoAnnotation[annotation_schema]))
        llm_input = LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=get_video_annotation_class(annotation_schema))
        output = ask(llm_input, config)
        if output is None:
            print(f'Error while generating annotations for {video_id}, skipping')
            continue
        outputs[video_id] = output.segments
        with open(config.get_anno_path(video_id), 'w') as f:
            json.dump(output.dict()['segments'], f)
    return outputs

def aggregate_annotations(config: DatagenConfig, filter_func = lambda x: True):
    annotations = config.get_annotations()
    segments = config.get_segments()
    segments = {video_id: [s for s in segments if s.video_id==video_id] for video_id in set([s.video_id for s in segments])}

    annotations_agg = []
    for video_id, video_annotations in annotations.items():
        i = 0
        for ann in sorted(video_annotations, key=lambda x: x['start_timestamp']):
            if not [s for s in segments[video_id] if s.start_timestamp==ann['start_timestamp'] and s.end_timestamp==ann['end_timestamp']]:
                # segment timestamps do not correspond to a segment and were hallucinated
                print('skipping', video_id)
                continue
            if not filter_func(ann['segment_annotation']):
                continue
            ann['video_id'] = video_id
            ann['id'] = f'{video_id}_{i}'
            ann['video_path'] = f'{video_id}_{i}.mp4'
            annotations_agg.append(ann)
            i+=1
    return annotations_agg
    

