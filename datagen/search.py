from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm
import yt_dlp
import pandas as pd

from .core.chat import ask
from .core.config import DatagenConfig
from .core.types import LLMInput


default_prompt = 'I want to find 10000 videos of people doing squats. The videos have to have some kind of explanations about how to do them correctly or how to not do them incorrectly or feedback on how they are done. Generate 100 search queries.'

# class Query(BaseModel):
#     'A youtube search query'
#     query: str = 

class QueryList(BaseModel):
    """A list of queries to find videos on a video hosting service"""
    queries: List[str] = Field(default=None, description="a list of queries")


def get_queries(config: DatagenConfig, prompt=default_prompt, num_queries=10):
    system_prompt = f'You a helping the user to find a very large and diverse set of videos on a video hosting service. A user will only describe which videos they are looking for and how many queries they need.'
    num_queries_prompt = f'I need {num_queries} queries'
    res = ask(llm_input=LLMInput(human_prompt=[prompt,num_queries_prompt], system_prompt=system_prompt, output_schema=QueryList), config=config)
    return res and res.queries


def get_video_info(query_list, videos_per_query=100):
    queries = {}
    # 'format': 'bestaudio', 'noplaylist':'True', 
    YDL_OPTIONS = {
    # "quiet":    True,
    "simulate": True,
    "forceurl": True,
    }

    for query in tqdm(query_list):
        if query not in queries:
            with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
                videos = ydl.extract_info(f"ytsearch{videos_per_query}:{query}", download=False)['entries']
            queries[query] = videos

    q = []
    for k,v in queries.items():
        for vv in v:
            vv['query'] = k
            q.append(vv)
    df = pd.DataFrame(q)
    df = agg_df(df)
    return df

def agg_video(dfv: pd.DataFrame):
    item = dfv.iloc[0]
    queries = []
    for q in dfv['query']:
        if type(q)==str:
            q = [q]
        queries.append(q)
    item['queries'] = queries
    return item

def agg_df(df: pd.DataFrame):
    df_unique = df.groupby('id').apply(agg_video)
    return df_unique