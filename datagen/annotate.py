from typing import Optional
import json

from tqdm import tqdm
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

def generate_annotations(
        annotation_schema: type[BaseModel],
        config: DatagenConfig,
        video_ids: Optional[list[str]] = None,
        filter_by: Optional[str] = None,
        system_prompt: str = system_prompt_default):
    
    if video_ids is None:
        video_ids = config.get_video_ids()

    processed_video_ids = [x.stem for x in config.anno_dir.iterdir()]

    segments = config.get_segments(video_ids=video_ids)

    outputs = {}
    for video_id in tqdm(set(video_ids) - set(processed_video_ids)):
        print(video_id, '- started')
        # if config.get_anno_path(video_id).exists():
        #     print(f'Annotation {video_id} exists, skipping.')
        #     continue
        video_segments = [s for s in segments if s.video_id==video_id]
        if filter_by:
            video_segments = [s for s in video_segments if s.segment_info[filter_by]]
        
        prompt = []
        prompt.append('Segment information:\n' + '\n'.join([s.to_str(skip=[filter_by] if filter_by else []) for s in video_segments]))
        transcript: str = config.get_transcript(video_id)
        if transcript:
            prompt.append('Transcript:\n' + transcript)
        llm_input = LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=get_video_annotation_class(annotation_schema))
        output = ask(llm_input, config)
        if output is None:
            print(f'Error while generating annotations for {video_id}, skipping')
            continue
        outputs[video_id] = output.segments
        with open(config.get_anno_path(video_id), 'w') as f:
            json.dump(output.dict()['segments'], f)
        print(video_id, '- done')
    return outputs

def aggregate_annotations(config: DatagenConfig, filter_func = lambda x: True, annotation_file='annotations.json'):
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
    config.dump(annotations_agg, config.data_dir / annotation_file)
    return annotations_agg
    