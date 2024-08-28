import decord
import time
from pathlib import Path
from collections import defaultdict
from datagen.detect_segments import get_segments
import torch
from transformers import AutoModel, AutoProcessor
import pandas as pd
from tsmoothie.smoother import LowessSmoother

from typing import List

from langchain.pydantic_v1 import BaseModel, Field

# decord.bridge.set_bridge("torch")

from .scraping import VideoInfo


class SegmentInfo(BaseModel):
    start_timestamp: str
    end_timestamp: str
    fps: float
    video_id: str


class VideoInferenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_infos: List[VideoInfo], local_root: Path):
        super(VideoInferenceDataset).__init__()

        self.video_infos = video_infos
        self.local_root = local_root
        self.frame_generator = self.get_frame_generator(video_infos, local_root)

    @staticmethod
    def get_frame_generator(video_infos, local_root: Path):

        for video_idx, video_info in enumerate(video_infos):
            video_path = local_root.joinpath(video_info.relative_video_path)
            vr = decord.VideoReader(str(video_path))
            num_frames = len(vr)
            fps = vr.get_avg_fps()
            frame_indices = range(0, num_frames, round(fps))

            for frame_idx in frame_indices:
                # print(f"Frame idx {frame_idx}")
                frame = vr[frame_idx].asnumpy()
                yield {
                    "frame": frame,
                    "frame_idx": frame_idx,
                    "video_id": video_idx,
                }

    def __next__(self):
        return next(self.frame_generator)

    def __iter__(self):
        return self


def detect_segments(
    video_infos: List[VideoInfo], clip_text_prompts: List[str]
) -> List[SegmentInfo]:

    LOCAL_ROOT = Path("./tmp/agent_squats").resolve()
    CLIP_MODEL_ID = "google/siglip-so400m-patch14-384"

    model = AutoModel.from_pretrained(CLIP_MODEL_ID).to("cuda")
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)

    dataset = VideoInferenceDataset(video_infos, LOCAL_ROOT)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=12,
        pin_memory=True,
        # worker_init_fn=worker_init_fn,
    )
    dataloader = iter(dataloader)

    smoother = LowessSmoother(smooth_fraction=0.02, iterations=1)

    clip_results_dict = defaultdict(list)

    print("Init model complete")

    batch_counter = 0
    MAX_BATCHES = 50

    while batch_counter < MAX_BATCHES:
        batch_counter += 1
        try:
            start_time = time.time()
            batch = next(dataloader)
            # print(f"Fetch time: {time.time() - start_time:.2f} seconds")
        except StopIteration:
            break

        inputs = processor(
            images=batch["frame"],
            text=clip_text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = model(**inputs)

        logits = outputs.logits_per_image
        probs = torch.nn.functional.sigmoid(logits).detach().cpu().numpy()

        for video_idx, frame_idx, prob in zip(
            batch["video_id"], batch["frame_idx"], probs
        ):
            # print(type(video_id.item()), type(frame_idx.item()), type(prob.item()))
            video_id = video_infos[video_idx.item()].video_id

            clip_results_dict["video_id"].append(video_id)
            clip_results_dict["frame_idx"].append(frame_idx.item())
            clip_results_dict["probs"].append(prob.item())

    print("All frames processed")
    clip_results = pd.DataFrame(clip_results_dict)
    print("Dataframe created")
    print(clip_results)

    max_gap_seconds = 1
    fps_sampling = 1
    min_prob = 0.1
    min_segment_seconds = 3
    fps = 25

    segment_infos = []
    for video_id, video_clip_results in clip_results.groupby("video_id"):
        probs = video_clip_results["probs"].values
        probs = smoother.smooth(probs).smooth_data[0]
        segments_start_end = get_segments(
            probs,
            max_gap=round(max_gap_seconds * fps_sampling),
            min_prob=min_prob,
            min_segment=round(min_segment_seconds * fps_sampling),
        )

        print(f"Segments for video {video_id}: {segments_start_end}")

        sec2ts = lambda s: time.strftime(
            f"%H:%M:%S.{round((s%1)*1000):03d}", time.gmtime(s)
        )

        for start, end in segments_start_end:
            segment_infos.append(
                SegmentInfo(
                    start_timestamp=sec2ts(start),
                    end_timestamp=sec2ts(end),
                    fps=fps,
                    video_id=video_id,
                )
            )

    return segment_infos
