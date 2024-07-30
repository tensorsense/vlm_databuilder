from typing import Optional
import json

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
                1. You can only rely on data presented in a provided JSON object. Don't improvise
                2. Follow user's request carefully
                3. Don't rush to deliver the answer. Take some time to think. Make a deep breath. Then start writing.
'''

def generate_annotations(
        human_prompt: str,
        annotation_schema: type[BaseModel],
        config: DatagenConfig,
        video_ids: Optional[list[str]] = None,
        # filter_by: Optional[str] = None,
        system_prompt: str = system_prompt_anno
        ):

    if video_ids is None:
        video_ids = config.get_video_ids()

    processed_video_ids = [x.stem for x in config.anno_dir.iterdir()]

    clues = config.get_clues(video_ids=video_ids)

    outputs = {}
    for video_id in tqdm((set(video_ids) - set(processed_video_ids)) & set([k for k,v in clues.items() if v is not None])):
        print(video_id, '- started')
        # if config.get_anno_path(video_id).exists():
        #     print(f'Annotation {video_id} exists, skipping.')
        #     continue
        # video_segments = [s for s in segments if s.video_id==video_id]
        # if filter_by:
        #     video_segments = [s for s in video_segments if s.segment_info and s.segment_info[filter_by]]

        prompt = [human_prompt]

        if clues[video_id]:
            for clue in clues[video_id]:
                prompt.append('Segment:\n' + json.dumps(clue))
        llm_input = LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=get_video_annotation_class(annotation_schema))
        output = ask(llm_input, config)
        if output is None:
            print(f'Error while generating annotations for {video_id}, skipping')
            continue
        outputs[video_id] = output.segments
        with open(config.get_anno_path(video_id), 'w') as f:
            json.dump(output.dict()['segments'], f)
        print(video_id, '- done')

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
    



system_prompt_clues_default = '''You are a highly intelligent data investigator. 
You take unstructured messy data and look for clues that could help interpet this
data in the right way.
You are the best one for this job in the world because you are a former detective. 
You care about even the smallest details. 
You use deductive and inductive reasoning at the highest possible quality.
#YOUR TODAY'S JOB
The user needs to guess about what happens on a specific *part* of a video file. Your job is to help the user by
providing clues that would help the user make the right assumption. The user will provide you: 
1. A list of time codes of the *parts* in format "<HH:MM:SS.ms>-<HH:MM:SS.ms>". The timecode is your starting point. Your logic will be mostly driven by timecodes. 
2. Data about what supposedly happens in each *part* and what kind of information the user is trying to obtain.
3. A transcript of the *full video* in format of "<HH.MM.SS>\\n<text>"
 
Your task:
1. Read the transcript.
2. Provide the clues in a given schema.
#RULES
!!! VERY IMPORTANT !!!
1. Rely only on the data provided in the transcript. Do not improvise.
2. Your job is to find the data already provided in the transcript.
3. Follow the schema output.
4. Be very careful with details. Don't generalize. Keep all the terms.
You always double check your results.
'''

def generate_clues(
        annotation_schema: type[BaseModel],
        config: DatagenConfig,
        segments_per_call: Optional[int] = 5,
        video_ids: Optional[list[str]] = None,
        filter_by: Optional[str] = None,
        system_prompt: str = system_prompt_clues_default,
        human_prompt: Optional[str] = None    
    ):

    if video_ids is None:
        video_ids = config.get_video_ids()

    processed_video_ids = [x.stem for x in config.clues_dir.iterdir()]

    segments = config.get_segments(video_ids=video_ids)

    outputs = {}
    for video_id in tqdm((set(video_ids) - set(processed_video_ids)) & set([s.video_id for s in segments])):
        print(video_id, '- started')
        # if config.get_anno_path(video_id).exists():
        #     print(f'Annotation {video_id} exists, skipping.')
        #     continue
        video_segments = [s for s in segments if s.video_id==video_id]
        if filter_by:
            video_segments = [s for s in video_segments if s.segment_info and s.segment_info[filter_by]]
        outputs[video_id] = []
        
        if segments_per_call:
            segments_array = [video_segments[i:i+segments_per_call] for i in range(0, len(video_segments), segments_per_call)]
        else:
            segments_array = video_segments
        for i, video_segments_part in enumerate(segments_array):
            print(f'{video_id} part {i} - started')
            prompt = []
            if human_prompt:
                prompt.append(human_prompt)
            prompt.append('Segment information:\n' + '\n'.join([s.to_str(skip=[filter_by] if filter_by else []) for s in video_segments_part]))
            transcript: str = config.get_transcript(video_id)
            if transcript:
                prompt.append('Transcript:\n' + transcript)
            llm_input = LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=get_video_annotation_class(annotation_schema))
            output = ask(llm_input, config)
            if output is None:
                print(f'Error while generating annotations for {video_id}, skipping')
                continue
            outputs[video_id].extend(output.segments)
        with open(config.get_clues_path(video_id), 'w') as f:
            json.dump([x.dict() for x in outputs[video_id]], f)
        print(video_id, '- done')
    return outputs