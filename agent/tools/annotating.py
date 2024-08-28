from typing import List, Optional
from collections import defaultdict
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# 4. Create nodes

from .scraping import VideoInfo
from .video_chunking import SegmentInfo


class LocalClue(BaseModel):
    """Local clues for a segment"""

    id: str = Field(description="LC1,LC2...")
    quote: str = Field(
        description="the quote from the transcript that was used to create this clue."
    )
    quote_timestamp_start: str = Field(
        description="the exact start timestamp of the quote."
    )
    quote_timestamp_end: str = Field(
        description="the exact end timestamp of the quote."
    )
    clue: str = Field(description="the main clue data")


class GlobalClue(BaseModel):
    """Global clues for a segment"""

    id: str = Field(description="GC1,GC2...")
    quote: str = Field(
        description="the quote from the transcript that was used to create this clue."
    )
    quote_timestamp_start: str = Field(
        description="the exact start timestamp of the quote."
    )
    quote_timestamp_end: str = Field(
        description="the exact end timestamp of the quote."
    )
    clue: str = Field(description="the main clue data.")
    relevance_to_segment: str = Field(
        description="why do you think this global clue is relevant to the segment you are working with right now."
    )


class LogicalInference(BaseModel):
    """Logical inferences for a segment"""

    id: str = Field(description="LI1,LI2,...")
    description: str = Field(description="A concise form of the logical inference.")
    details: str = Field(
        description="A verbose explanation of what insight about what happens in this segment should be made based on the clues that you found."
    )


class SegmentAnnotation(BaseModel):
    local_clues: list[LocalClue] = Field(
        description="Local clues are inside the segment in terms of timestamps."
    )
    global_clues: list[GlobalClue] = Field(
        description="Global clues are scattered across the entire transcript."
    )
    logical_inferences: list[LogicalInference] = Field(
        description="What can we infer about the topic, that the user is looking for in the video, can we make based on the clues inside this segment"
    )


class SegmentWithClueInfo(BaseModel):
    """
    Annotation for a video segment.
    """

    start_timestamp: str = Field(
        description="start timestamp of the segment in format HH:MM:SS.MS"
    )
    end_timestamp: str = Field(
        description="start timestamp of the segment in format HH:MM:SS.MS"
    )
    segment_annotation: SegmentAnnotation = Field(
        description="list of annotations for the segment"
    )


class VideoAnnotation(BaseModel):
    """
    Segments of a video.
    """

    segments: list[SegmentWithClueInfo] = Field(
        description="information about each segment"
    )


def extract_clues(
    llm,
    system_prompt: str,
    segment_infos: List[SegmentInfo],
    video_infos: List[VideoInfo],
):

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "Segment timecodes: {{ segment_timecodes }}\nTranscript: {{ transcript }}",
            ),
        ],
        template_format="jinja2",
    )

    model = prompt_template | llm.with_structured_output(VideoAnnotation)

    segment_infos_dict = defaultdict(list)
    for segment_info in segment_infos:
        segment_infos_dict[segment_info.video_id].append(segment_info)

    video_infos_dict = {video_info.video_id: video_info for video_info in video_infos}

    clues = []

    for video_id, segment_infos in segment_infos_dict.items():
        transcript = video_infos_dict[video_id].transcript
        segment_infos_chunks = [
            segment_infos[i : i + 5] for i in range(0, len(segment_infos), 5)
        ]

        for chunk in segment_infos_chunks:
            video_annotation: VideoAnnotation = model.invoke(
                {
                    "segment_timecodes": "\n".join(
                        [f"{s.start_timestamp}-{s.end_timestamp}" for s in chunk]
                    ),
                    "transcript": transcript,
                }
            )
            clues.extend(video_annotation.segments)

    return clues


def gen_annotations(llm, system_prompt: str, clues: List[SegmentAnnotation]):
    class SegmentFeedback(BaseModel):
        right: Optional[str] = Field(description="what was right in the performance")
        wrong: Optional[str] = Field(description="what was wrong in the performance")
        correction: Optional[str] = Field(
            description="how and in what ways it the performance could be improved"
        )

    # The segment timestamps are taken from the provided information.
    class SegmentCompleteAnnotation(BaseModel):
        squats_probability: Optional[str] = Field(
            description="how high is the probability that the person is doing squats in the segment: low, medium, high, unknown(null)"
        )
        squats_technique_correctness: Optional[str] = Field(
            description="correctness of the squat technique."
        )
        squats_feedback: Optional[SegmentFeedback] = Field(
            description="what was right and wrong in the squat perfomance in the segment. When the technique is incorrect, provide instructions how to correct them."
        )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Clues: {{ clues }}"),
        ],
        template_format="jinja2",
    )

    model = prompt_template | llm.with_structured_output(SegmentCompleteAnnotation)

    annotations = []
    for clue in clues:
        segment_annotation: SegmentCompleteAnnotation = model.invoke(
            {"clues": clue.json()}
        )

        annotations.append(segment_annotation.json())

    return annotations
