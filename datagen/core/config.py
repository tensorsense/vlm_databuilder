from pathlib import Path
import json

from pydantic import BaseModel, Field, DirectoryPath
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import yaml
import dotenv

from .sub_utils import vtt_to_txt
from .types import Segment

dotenv.load_dotenv()

class OpenAIConfig(BaseModel):
    temperature: float
    deployment: str

    def __init__(self, **data):
        super().__init__(**data)
        if data['type'] == 'azure':
            openai_class = AzureChatOpenAI
        elif data['type'] == 'openai':
            openai_class = ChatOpenAI
        else:
            raise ValueError('config.openai.type should be "openai" or "azure"')
        self._model = openai_class(temperature=self.temperature, model=self.deployment)


class DatagenConfig(BaseModel):
    data_dir: DirectoryPath

    openai: OpenAIConfig

    def __init__(self, **data):
        super().__init__(**data)

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.sub_dir.mkdir(parents=True, exist_ok=True)
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        

    @property
    def model(self) -> AzureChatOpenAI:
        return self.openai._model

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            yaml_config = yaml.safe_load(f)

        yaml_config['data_dir'] = Path(yaml_config['data_dir'])
        yaml_config['data_dir'].mkdir(parents=True, exist_ok=True)
        config = cls(**yaml_config)

        return config
    
    @property
    def value(self) -> int:
        return 1
    
    @property
    def video_dir(self):
        return self.data_dir / 'videos'
    
    @property
    def sub_dir(self):
        return self.data_dir / 'subs'

    @property
    def segment_dir(self):
        return self.data_dir / 'segments'

    @property
    def clip_dir(self):
        return self.data_dir / 'clips'

    @property
    def anno_dir(self):
        return self.data_dir / 'annotations'
    
    def get_video_path(self, video_id: str):
        return self.video_dir / f'{video_id}.mp4'
    
    def get_videos(self) -> list[Path]:
        return [v for v in self.video_dir.iterdir() if v.suffix=='.mp4']
    
    def get_sub_path(self, video_id: str):
        return self.sub_dir / f'{video_id}.en.vtt'
    
    def get_transcript(self, video_id: str) -> str:
        if self.get_sub_path(video_id).exists():
            return vtt_to_txt(self.get_sub_path(video_id))
    
    def get_segment_path(self, video_id: str):
        return self.segment_dir / f'{video_id}.json'

    def save_segments(self, video_id: str, segments: list[Segment]):
        with open(self.get_segment_path(video_id), 'w') as f:
            json.dump([s.dict() for s in segments], f)

    def get_segments(self, info_type: type[BaseModel] = BaseModel):
        segments = []
        for segment_info_path in self.segment_dir.iterdir():
            with open(segment_info_path) as f:
                segments.extend([Segment[info_type].parse_obj(s) for s in json.load(f)])
        return segments
    
    def get_anno_path(self, video_id: str):
        return self.anno_dir / f'{video_id}.json'
    
    def get_annotations(self) -> dict[list, object]:
        annotations = {}
        for anno_path in self.anno_dir.iterdir():
            video_id = anno_path.stem
            if video_id not in annotations:
                annotations[video_id] = []
            with open(anno_path) as f:
                annotations[video_id].extend(json.load(f))
        return annotations
