from typing import Optional
import json

from tqdm import tqdm
from langchain.pydantic_v1 import BaseModel, create_model, Field

from .core.types import Segment, LLMInput
from .core.config import DatagenConfig
from .core.chat import ask
from .core.types import get_video_annotation_class


# human_prompt = """User's instructions:
# The initial video was a tutorial about how to perform squats. 
# All *parts* below contain a video footage of a person doing squats. 
# I need to find as much data as possible about HOW THIS PERSON PERFORMS SQUATS. 
# I'm interested in how a person in a segment doings squats. What mistakes they make. What improvements they show. 
# What they do correctly. What could be improved.
# Please, help me find relevant clues.
# """

from datagen import ask, LLMInput
local_clues_default_examples =    '''
        Good local clues examples: [
      {
        "id": "LC1",
        "timestamp": "00:00:19",
        "text": "exercises do them wrong and instead of",
        "analysis": "This phrase introduces the concept of incorrect exercise form, setting the stage for a demonstration of improper technique."
      },
      {
        "id": "LC2",
        "timestamp": "00:00:21",
        "text": "growing nice quads and glutes you'll",
        "analysis": "Mentions the expected benefits of proper squats (muscle growth), implying that these benefits won't be achieved with incorrect form."
      },
      {
        "id": "LC3",
        "timestamp": "00:00:22",
        "text": "feel aches and pains in your knees your",
        "analysis": "Directly states negative consequences of improper form, strongly suggesting that this segment demonstrates incorrect technique."
      },
      {
        "id": "LC4",
        "timestamp": "00:00:24",
        "text": "lower back and even your shoulders",
        "analysis": "Continuation of LC3, emphasizing multiple areas of potential pain from improper form."
      },
      {
        "id": "LC5",
        "timestamp": "00:00:26",
        "text": "let's see how to do it correctly",
        "analysis": "This phrase suggests a transition is about to occur. The incorrect form has been shown, and correct form will follow."
      }
    ]
'''
global_clues_default_examples = '''
    Good global clues examples: [
      {
        "id": "GC1",
        "timestamp": "00:00:08",
        "text": "the most common mistake",
        "analysis": "Introduces the idea that a frequent error will be discussed. This sets up the expectation for a demonstration of this mistake."
      },
      {
        "id": "GC2",
        "timestamp": "00:00:10",
        "text": "is when your heels are",
        "analysis": "Begins to describe the specific error related to heel position."
      },
      {
        "id": "GC3",
        "timestamp": "00:00:12",
        "text": "in the air and not attached to the ground",
        "analysis": "Completes the description of the common mistake. This strongly suggests that the segment will demonstrate this specific error."
      },
      {
        "id": "GC4",
        "timestamp": "00:01:01",
        "text": "butt wink is a problem",
        "analysis": "Introduces another potential issue in squat form. While this comes after the segment, it might be relevant if the demonstration includes multiple errors."
      },
      {
        "id": "GC5",
        "timestamp": "00:01:03",
        "text": "it can lead to the back pain",
        "analysis": "Connects to LC3 and LC4, which mention back pain. This strengthens the possibility that 'butt wink' is also demonstrated in the segment."
      },
      {
        "id": "GC6",
        "timestamp": "00:01:06",
        "text": "so don't do that",
        "analysis": "Reinforces that the previously mentioned 'butt wink' is an error to be avoided, consistent with the segment's focus on incorrect form."
      }
    ]
'''

logic_default_examples = '''
    Good logical inference examples:
    [
      {
        "id": "LI1",
        "description": "Primary Demonstration of Heel Lift",
        "details": "Given that GC1-GC3 describe the 'most common mistake' as heels lifting off the ground, and this description immediately precedes our segment, it's highly probable that this is the primary error being demonstrated. This is further supported by the segment's focus on incorrect form (LC1-LC4)."
      },
      {
        "id": "LI2",
        "description": "Multiple Error Demonstration",
        "details": "While heel lift is likely the primary focus, the mention of multiple pain points (knees, lower back, shoulders in LC3-LC4) suggests that the demonstrator may be exhibiting several forms of incorrect technique simultaneously. This comprehensive 'what not to do' approach would be pedagogically effective."
      },
      {
        "id": "LI3",
        "description": "Possible Inclusion of 'Butt Wink'",
        "details": "Although 'butt wink' is mentioned after our segment (GC4-GC6), its connection to back pain (which is mentioned in LC4) raises the possibility that this error is also present in the demonstration. The instructor may be showing multiple errors early on, then breaking them down individually later."
      },
      {
        "id": "LI4",
        "description": "Segment Placement in Overall Video Structure",
        "details": "The segment's position (starting at 00:00:19) and the phrase 'let's see how to do it correctly' (LC5) at the end suggest this is an early, foundational part of the video. It likely serves to grab attention by showing common mistakes before transitioning to proper form instruction."
      },
      {
        "id": "LI5",
        "description": "Intentional Exaggeration of Errors",
        "details": "Given the educational nature of the video, it's plausible that the demonstrator is intentionally exaggerating the incorrect form. This would make the errors more obvious to viewers and enhance the contrast with correct form shown later."
      }
    ]
    
    Good additional observations examples: [
      {
        "id": "AO1",
        "description": "Absence of Technical Terms",
        "details": "The transcript uses lay terms ('nice quads and glutes') rather than technical anatomical language. This suggests the video is targeted at a general audience rather than fitness professionals."
      },
      {
        "id": "AO2",
        "description": "Emphasis on Consequences",
        "details": "The immediate focus on negative outcomes (pain, lack of muscle growth) indicates a motivational approach, likely to encourage viewers to pay close attention to form."
      },
      {
        "id": "AO3",
        "description": "Potential Visual Cues",
        "details": "While we can't see the video, the specific mentions of body parts (heels, knees, lower back, shoulders) suggest there may be visual indicators or graphics highlighting these areas during the demonstration."
      },
      {
        "id": "AO4",
        "description": "Instructional Flow",
        "details": "The structure (common mistake → demonstration of errors → transition to correct form) follows a classic 'what not to do, then what to do' instructional pattern, which is effective for physical skills."
      }
    ]
    '''

def generate_clues_dataclass(config: DatagenConfig, prompt: Optional[str]=None):
    # if prompt is None:
    local_clues_examples = local_clues_default_examples
#     else:
#         system_prompt_local_clues_dataclass = f'''
# You are given examples of json info objects that are extracted from video transcripts.
# The videos that these examples are used for are instructional videos for improving squat technique.
# Your task is to generate the same kinds of examples, but for videos of the kind: "{prompt}".
# Your output should be in exactly the same form at the user input, do not add any additional information.

# Local clues description:
# "Explain your logic in “If A then B” style. E.g., "Dan says Tony was doing squats right while Mary did it wrong, and according to the conversation the person in this segment is Tony".
# A clue is considered local if its located inside the segment or is overlapping with it. Be excessive, provide all the information you have found."
# '''
#         local_clues_examples = ask(llm_input=LLMInput(system_prompt=system_prompt_local_clues_dataclass, human_prompt=[local_clues_default_examples]), config=config)

    LocalClue = create_model(
        'LocalClue',
        id = (str, Field(description='LC1,LC2...')),
        quote = (str, Field(description='a direct quote taken from the transcript. The quote must be directly inside the segment, based on the timestamps in the transcript and the segment.')),
        quote_timestamp = (str, Field(description='The timestamp of the quote. Must be taken directly from the transcript. It is very important the quote is inside the segment, for example if the segment is 00:10:25-00:11:05, then the quotes could have timestamps of 00:10:40 or 00:11:00, but not 00:09:30 or 00:11:07.')),
        analysis = (str, Field(description='interpretation of the text for improving squat techique')),
    )
    LocalClue.__doc__ = local_clues_examples

    # if prompt is None:
    global_clues_examples = global_clues_default_examples
#     else:
#         system_prompt_global_clues_dataclass = f'''
# You are given examples of json info objects that are extracted from video transcripts.
# The videos that these examples are used for are instructional videos for improving squat technique.
# Your task is to generate the same kinds of examples, but for videos of the kind: "{prompt}".
# Your output should be in exactly the same form at the user input, do not add any additional information.

# Global clues description:
# "Global" means these clues were found across the entire video. E.g., the segment happens at 00:00:15 and the clue was found at 01:19:11. Explain your logic, especially why these clues are relevant to this particular segment. Be excessive, provide all the information you have found. Provide specific instructions from the transcript with timecodes."'''
#         global_clues_examples = ask(llm_input=LLMInput(system_prompt=system_prompt_global_clues_dataclass, human_prompt=[global_clues_default_examples]), config=config)

    GlobalClue = create_model(
        'GlobalClue',
        id = (str, Field(description='GC1,GC2...')),
        quote = (str, Field(description='a direct quote taken from the transcript that is referncing this specific segment of the video. The quote could be taken from anywhere in the transcript.')),
        quote_timestamp = (str, Field(description='The timestamp of the quote. Must be taken directly from the transcript. Could be taken from anywhere in the video, not necessarily inside the segment, for example the segment happens at 00:00:15-00:02:23 and the clue was found at 01:19:11')),
        analysis = (str, Field(description='interpretation of the text for improving squat techique')),
    )
    GlobalClue.__doc__ = global_clues_examples

    # if prompt is None:
    logic_examples = logic_default_examples
#     else:
#         system_prompt_logic_dataclass = f'''
# You are given examples of json info objects that are extracted from video transcripts.
# The videos that these examples are used for are instructional videos for improving squat technique.
# Your task is to generate the same kinds of examples, but for videos of the kind: "{prompt}".
# Your output should be in exactly the same form at the user input, do not add any additional information.

# Logical inferences description: Build logical inferences for clues you found before. Use technical language. Be clear and consistent.
# Additional Observations description: Any other observations that could help interpret the part of the video.'''
#         logic_examples = ask(llm_input=LLMInput(system_prompt=system_prompt_logic_dataclass, human_prompt=[logic_default_examples]), config=config)

    AdditionalInformation = create_model(
        'AdditionalInformation',
        id = (str, Field(description='LI1,LI2,... for logical inference, AO1,AO2,... for additional observations.')),
        description = (str, Field(description='A concise name of the information')),
        details = (str, Field(description='a more verbose description related to improving squat technique')),
    )
    AdditionalInformation.__doc__ = logic_examples

    SegmentAnnotation = create_model(
        'SegmentAnnotation',
        local_clues = (Optional[list[LocalClue]], Field(description=f'Provide here all the clues about this time segment{None if prompt is None else ". The clues must be relevant for " + prompt}. Explain your logic in “If A then B” style. E.g., "Dan says Tony was doing squats right while Mary did it wrong, and according to the conversation the person in this segment is Tony". Local clues must located inside the segment, for example if the segment is 00:10:25-00:11:05, then a local clues could be found at 00:10:40 or 00:11:00. Be excessive, provide all the information you have found. Provide specific instructions from the transcript with timecodes.')),
        global_clues = (Optional[list[GlobalClue]], Field(description=f'Relevant clues are also scattered across the entire video{None if prompt is None else ". The clues must be relevant for " + prompt}. Provide here all the global clues about this time segment. Global clues can be found across the entire video. E.g., the segment happens at 00:00:15 and the clue was found at 01:19:11. Explain your logic, especially why these clues are relevant to this particular segment. Be excessive, provide all the information you have found. Provide specific instructions from the transcript with timecodes.')),
        logical_inferences = (Optional[list[AdditionalInformation]], Field(description='Build logical inferences for clues you found before. Use technical language. Be clear and consistent.')),
        additional_observations = (Optional[list[AdditionalInformation]], Field(description='Any other observations that could help interpret the part of the video.'))
    )

    return SegmentAnnotation


# system_prompt_clues_default = '''You are a highly intelligent data investigator. 
# You take unstructured messy data and look for clues that could help interpet this
# data in the right way.
# You are the best one for this job in the world because you are a former detective. 
# You care about even the smallest details. 
# You use deductive and inductive reasoning at the highest possible quality.
# #YOUR TODAY'S JOB
# The user needs to guess about what happens on a specific *part* of a video file. Your job is to help the user by
# providing clues that would help the user make the right assumption. The user will provide you: 
# 1. A list of time codes of the *parts* in format "<HH:MM:SS.ms>-<HH:MM:SS.ms>". The timecode is your starting point. Your logic will be mostly driven by timecodes. 
# 2. Data about what supposedly happens in each *part* and what kind of information the user is trying to obtain.
# 3. A transcript of the *full video* in format of "<HH.MM.SS>\\n<text>"
 
# Your task:
# 1. Read the transcript.
# 2. Provide the clues in a given schema.
# #RULES
# !!! VERY IMPORTANT !!!
# 1. Rely only on the data provided in the transcript. Do not improvise.
# 2. Your job is to find the data already provided in the transcript.
# 3. For the local clues, make sure that the quotes are inside the segment or mostly overlap with the segment. To do this, double check the timestamps from the transcript and the segment.
# 4. Follow the schema output.
# 5. Be very careful with details. Don't generalize. Keep all the terms.
# You always double check your results.
# '''
system_prompt_clues_default = '''You are a highly intelligent data investigator.  
You take unstructured damaged data and look for clues that could help restore the initial information
and extract important insights from it.
You are the best one for this job in the world because you are a former detective. 
You care about even the smallest details, and your guesses about what happened in the initial file
even at very limited inputs are usually absolutely right.  
You use deductive and inductive reasoning at the highest possible quality.

#YOUR TODAY'S JOB
The user needs to guess about what happens on a specific *part* of a video file. Your job is to help the user by
providing clues that would help the user make the right assumption. The user will provide you: 
1. A list of time codes of the *parts* in format "<HH:MM:SS.ms>-<HH:MM:SS.ms>". The timecode is your starting point. Your logic will be mostly driven by timecodes. 
2. Instructions about what kind of information the user is trying to obtain.
3. A transcript of the *full video* in format of "<HH.MM.SS>\\n<text>"
 
Your task:
1. Read the transcript.
2. Provide the clues in a given format.
3. Provied any other info requested by the user.

#RULES
!!! VERY IMPORTANT !!!
1. Rely only on the data provided in the transcript. Do not improvise.
2. Your job is to find the data already provided in the transcript.
3. Analyze every segment. Only skip a segment if there is no information about it in the trascript.
4. For local clues, make sure that the quotes are inside the segment or mostly overlap with the segment. To do this, double check the timestamps from the transcript and the segment.
5. Follow the format output.
6. Be very careful with details. Don't generalize. Always double check your results.
'''



def generate_clues(
        annotation_schema: type[BaseModel],
        config: DatagenConfig,
        segments_per_call: Optional[int] = 5,
        video_ids: Optional[list[str]] = None,
        filter_by: Optional[str] = None,
        system_prompt: str = system_prompt_clues_default,
        human_prompt: Optional[str] = None,
        raise_on_error: bool = False,
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
                print(f'Error while generating annotations for {video_id} part {i}, skipping')
                if raise_on_error:
                    raise Exception('exception in gpt call, exiting.')
                continue
            outputs[video_id].extend(output.segments)
        with open(config.get_clues_path(video_id), 'w') as f:
            json.dump([x.dict() for x in outputs[video_id]], f)
        print(video_id, '- done')
    return outputs