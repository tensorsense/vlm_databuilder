from datetime import datetime
from typing import Generator, Literal, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm
import yt_dlp
# import pandas as pd
import scrapetube

from .core.chat import ask
from .core.config import DatagenConfig
from .core.types import LLMInput


default_prompt = 'I want to find 10000 videos of people doing squats. The videos have to have some kind of explanations about how to do them correctly or how to not do them incorrectly or feedback on how they are done. Generate 100 search queries.'

# class Query(BaseModel):
#     'A youtube search query'
#     query: str = 

class QueryList(BaseModel):
    """A list of queries to find videos on a video hosting service"""
    queries: list[str] = Field(default=None, description="a list of queries")


def get_queries(config: DatagenConfig, prompt=default_prompt, num_queries=10):
    system_prompt = f'You a helping the user to find a very large and diverse set of videos on a video hosting service. A user will only describe which videos they are looking for and how many queries they need.'
    num_queries_prompt = f'I need {num_queries} queries'
    res = ask(llm_input=LLMInput(human_prompt=[prompt,num_queries_prompt], system_prompt=system_prompt, output_schema=QueryList), config=config)
    return res and res.queries

def get_video_ids(
    query: str|list[str],
    config: DatagenConfig,
    out_path = 'video_ids.json',
    videos_per_query: Optional[int] = None,
    sleep: float = 0,
    sort_by: Literal["relevance", "upload_date", "view_count", "rating"] = "relevance",
    results_type: Literal["video", "channel", "playlist", "movie"] = "video",
    only_creative_commons=True,
) -> Generator[dict, None, None]:
    if type(query) is str:
        query = [query]
    ids = set()
    for q in tqdm(query):
        for video in scrapetube.get_search(query=q, limit=videos_per_query, sleep=sleep, sort_by=sort_by, results_type=results_type):
            ids.add(video['videoId'])
    ids = list(ids)

    if only_creative_commons:
        print(datetime.now(), 'Filtering ids that for creative commons license')
        ids_cc = []
        for i in tqdm(ids):
            YDL_OPTIONS = {
                "quiet":    True,
                "simulate": True,
                "forceurl": True,
            }
            with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
                info = ydl.extract_info(f"youtube.com/watch?v={i}", download=False)
            if 'creative commons' in info.get('license', '').lower():
                ids_cc.append(id)
        ids = ids_cc

    config.dump(ids, config.data_dir / out_path)
    return ids

# def get_video_info(query_list, videos_per_query=100):
#     queries = {}
#     # 'format': 'bestaudio', 'noplaylist':'True', 
#     YDL_OPTIONS = {
#     # "quiet":    True,
#     "simulate": True,
#     "forceurl": True,
#     }

#     for query in tqdm(query_list):
#         if query not in queries:
#             with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
#                 videos = ydl.extract_info(f"ytsearch{videos_per_query}:{query}", download=False)['entries']
#             queries[query] = videos

#     q = []
#     for k,v in queries.items():
#         for vv in v:
#             vv['query'] = k
#             q.append(vv)
#     df = pd.DataFrame(q)
#     df = agg_df(df)
#     return df

# def agg_video(dfv: pd.DataFrame):
#     item = dfv.iloc[0]
#     queries = []
#     for q in dfv['query']:
#         if type(q)==str:
#             q = [q]
#         queries.append(q)
#     item['queries'] = queries
#     return item

# def agg_df(df: pd.DataFrame):
#     df_unique = df.groupby('id').apply(agg_video)
#     return df_unique