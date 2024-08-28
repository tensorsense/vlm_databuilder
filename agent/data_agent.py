from typing import TypedDict, Annotated, Sequence, List, Optional
import operator
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.scraping import gen_queries, get_video_ids, download, VideoInfo
from tools.video_chunking import detect_segments, SegmentInfo
from tools.annotating import extract_clues, gen_annotations

from tools.prompts import (
    GEN_QUERIES_PROMPT,
    EXTRACT_CLUES_PROMPT,
    GEN_ANNOTATIONS_PROMPT,
)


llm = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt4o",
    openai_api_version="2023-07-01-preview",
)

memory = MemorySaver()
# memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    task: str
    search_queries: List[str]
    video_ids: List[str]
    video_infos: List[VideoInfo]
    clip_text_prompts: List[str]
    segment_infos: List[SegmentInfo]
    clues: List[str]
    annotations: List[str]


class DataAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("generate_queries", self.gen_queries_node)
        builder.add_node("get_video_ids", self.get_video_ids_node)
        builder.add_node("download", self.download_node)
        builder.add_node("detect_segments", self.detect_segments_node)
        builder.add_node("extract_clues", self.extract_clues_node)
        builder.add_node("gen_annotations", self.gen_annotations_node)

        builder.set_entry_point("generate_queries")

        builder.add_edge("generate_queries", "get_video_ids")
        builder.add_edge("get_video_ids", "download")
        builder.add_edge("download", "detect_segments")
        builder.add_edge("detect_segments", "extract_clues")
        builder.add_edge("extract_clues", "gen_annotations")
        builder.add_edge("gen_annotations", END)

        graph = builder.compile(checkpointer=memory)

        return graph

    def gen_queries_node(self, state: AgentState):
        search_queries = gen_queries(self.llm, state["task"], GEN_QUERIES_PROMPT)
        return {"search_queries": search_queries[:2]}

    def get_video_ids_node(self, state: AgentState):
        video_ids = get_video_ids(state["search_queries"])
        return {"video_ids": video_ids}

    def download_node(self, state: AgentState):
        video_infos = download(state["video_ids"])
        return {"video_infos": video_infos}

    def detect_segments_node(self, state: AgentState):
        segment_infos = detect_segments(
            state["video_infos"], state["clip_text_prompts"]
        )
        return {"segment_infos": segment_infos}

    def extract_clues_node(self, state: AgentState):
        clues = extract_clues(
            self.llm,
            EXTRACT_CLUES_PROMPT,
            state["segment_infos"],
            state["video_infos"],
        )
        return {"clues": clues}

    def gen_annotations_node(self, state: AgentState):
        annotations = gen_annotations(self.llm, GEN_ANNOTATIONS_PROMPT, state["clues"])
        return {"annotations": annotations}

    def run(self, task: str, thread_id: str):
        thread = {"configurable": {"thread_id": thread_id}}
        for step in self.graph.stream(
            {
                "task": task,
                "clip_text_prompts": ["person doing squats"],
            },
            thread,
        ):
            if "download" in step:
                print("dowload happened")
            elif "extract_clues" in step:
                print("extract_clues happened")
            else:
                print(step)
