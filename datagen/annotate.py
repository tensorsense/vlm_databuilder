from typing import Optional
import json
from datetime import datetime

from tqdm import tqdm
from langchain.pydantic_v1 import BaseModel

from .core.types import Segment, LLMInput
from .core.config import DatagenConfig
from .core.chat import ask
from .core.types import get_video_annotation_class

system_prompt_anno = '''You are a helpful assistant that performs high quality data investigation and transformation.
                You will be given a JSON object with clues and other helpful information about what's going on 
                in a specific part of a video file. This part is called a segment. Your job is to:
                1. Read this JSON object carefully
                2. Answer user's questions about this segment
                3. Provide the answer as a JSON object in a schema provided by the user
                Important rules:
                1. You can only rely on data presented in a provided JSON object. Don't improvise.
                2. Follow user's request carefully.
                3. Don't rush to deliver the answer. Take some time to think. Make a deep breath. Then start writing.
                4. If you want to output field as empty (null), output it as JSON null (without quotes), not as a string "null". 
'''

def generate_annotations(
        annotation_schema: type[BaseModel],
        config: DatagenConfig,
        human_prompt: Optional[str] = None,
        video_ids: Optional[list[str]] = None,
        # filter_by: Optional[str] = None,
        system_prompt: str = system_prompt_anno,
        raise_on_error: bool = False,
        segments_per_call: Optional[int] = 10,
    ):

    if video_ids is None:
        video_ids = config.get_video_ids()

    processed_video_ids = [x.stem for x in config.anno_dir.iterdir()]

    clues = config.get_clues(video_ids=video_ids)

    outputs = {}
    for video_id in tqdm((set(video_ids) - set(processed_video_ids)) & set([k for k,v in clues.items() if v is not None])):
        print(datetime.now(), video_id, '- started')
        # if config.get_anno_path(video_id).exists():
        #     print(datetime.now(), f'Annotation {video_id} exists, skipping.')
        #     continue
        # video_segments = [s for s in segments if s.video_id==video_id]
        # if filter_by:
        #     video_segments = [s for s in video_segments if s.segment_info and s.segment_info[filter_by]]
        if not clues[video_id]:
            clues_array = []
        elif segments_per_call:
            clues_array = [clues[video_id][i:i+segments_per_call] for i in range(0, len(clues[video_id]), segments_per_call)]
        else:
            clues_array = clues[video_id]

        for i, clues_part in enumerate(clues_array):
            prompt = []
            if human_prompt:
                prompt.append(human_prompt)
            for clue in clues_part:
                prompt.append('Segment:\n' + json.dumps(clue))
            llm_input = LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=get_video_annotation_class(annotation_schema))
            output = ask(llm_input, config)
            if output is None:
                print(datetime.now(), f'Error while generating annotations for {video_id} part {i}, skipping')
                if raise_on_error:
                    raise Exception('exception in gpt call, exiting.')
                continue
            outputs[video_id] = output.segments
        if output is not None:
            with open(config.get_anno_path(video_id), 'w') as f:
                json.dump(output.dict()['segments'], f)
        else:
            print(datetime.now(), 'output is None, skipping.')
        print(datetime.now(), video_id, '- done')

def aggregate_annotations(config: DatagenConfig, filter_func = lambda x: True, annotation_file='annotations.json'):
    annotations = config.get_annotations()
    segments = config.get_segments()
    segments = {video_id: [s for s in segments if s.video_id==video_id] for video_id in set([s.video_id for s in segments])}

    annotations_agg = []
    for video_id, video_annotations in tqdm(annotations.items()):
        i = 0
        for ann in sorted(video_annotations, key=lambda x: x['start_timestamp']):
            if (video_id not in segments) or not [s for s in segments[video_id] if s.start_timestamp==ann['start_timestamp'] and s.end_timestamp==ann['end_timestamp']]:
                # segment timestamps do not correspond to a segment and were hallucinated
                print(datetime.now(), 'skipping', video_id)
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
