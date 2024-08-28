from typing import List

import scrapetube
import yt_dlp
from datetime import datetime
from pathlib import Path
from datagen.core.sub_utils import vtt_to_txt

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field


class VideoInfo(BaseModel):
    video_id: str
    url: str
    relative_video_path: str
    subs: str
    transcript: str


def gen_queries(llm, task: str, system_prompt: str) -> List[str]:
    class QueryList(BaseModel):
        """A list of queries to find videos on a video hosting service"""

        search_queries: list[str] = Field(default=None, description="a list of queries")

    messages = [
        SystemMessage(content=str(system_prompt)),
        HumanMessage(content=task),
    ]

    model = llm.with_structured_output(QueryList)
    response: QueryList = model.invoke(messages)

    return response.search_queries


def get_video_ids(queries: List[str]) -> List[str]:
    videos_per_query = 1
    sleep = 0
    sort_by = "relevance"
    results_type = "video"
    only_creative_commons = False

    video_ids = set()
    for query in queries:
        for video in scrapetube.get_search(
            query=query,
            limit=videos_per_query,
            sleep=sleep,
            sort_by=sort_by,
            results_type=results_type,
        ):
            video_ids.add(video["videoId"])
    video_ids = list(video_ids)

    if only_creative_commons:
        video_ids_cc = []
        for i in video_ids:
            YDL_OPTIONS = {
                "quiet": True,
                "simulate": True,
                "forceurl": True,
            }
            with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
                info = ydl.extract_info(f"youtube.com/watch?v={i}", download=False)
            if "creative commons" in info.get("license", "").lower():
                video_ids_cc.append(i)
        video_ids = video_ids_cc

    return video_ids


def download(video_ids: List[str]) -> List[VideoInfo]:

    LOCAL_ROOT = Path("./tmp/agent_squats").resolve()
    video_dir = LOCAL_ROOT / "videos"
    sub_dir = LOCAL_ROOT / "subs"

    discard_path = LOCAL_ROOT / "videos_without_subs"
    discard_path.mkdir(parents=True, exist_ok=True)

    downloaded_video_ids = [video_path.stem for video_path in video_dir.glob("*.mp4")]
    downloaded_video_ids += [
        video_path.stem for video_path in discard_path.glob("*.mp4")
    ]

    print(f"Downloaded video ids: {downloaded_video_ids}")

    only_with_transcripts = True

    YDL_OPTIONS = {
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "overwrites": False,
        "format": "mp4",
        "outtmpl": {
            "default": video_dir.as_posix() + "/%(id)s.%(ext)s",
            "subtitle": sub_dir.as_posix() + "/%(id)s.%(ext)s",
        },
    }

    video_infos = []

    with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"

            if video_id not in downloaded_video_ids:
                try:
                    ydl.download(url)
                except Exception as e:
                    print(datetime.now(), f"Error at video {video_id}, skipping")
                    print(datetime.now(), e)
                    continue

            video_path = Path(ydl.prepare_filename({"id": video_id, "ext": "mp4"}))
            sub_path = Path(
                ydl.prepare_filename(
                    {"id": video_id, "ext": "en.vtt"}, dir_type="subtitle"
                )
            )

            with sub_path.open("r") as f:
                subs = f.read()

            transcript = vtt_to_txt(sub_path)

            video_info = VideoInfo(
                video_id=video_id,
                url=url,
                relative_video_path=video_path.relative_to(LOCAL_ROOT).as_posix(),
                subs=subs,
                transcript=transcript,
            )

            video_infos.append(video_info)

    if only_with_transcripts:
        filtered_video_infos = []
        for video_info in video_infos:
            if video_info.transcript:
                filtered_video_infos.append(video_info)
            else:
                video_path = LOCAL_ROOT / video_info.video_path
                video_path.rename(discard_path / video_path.name)
        video_infos = filtered_video_infos

    return video_infos
