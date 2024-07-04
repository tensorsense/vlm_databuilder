# TensorSense Data Generation SDK

This SDK generates datasets for training Video LLMs from youtube videos. More sources coming later!

## 🐠 What it Does
- Generate search queries with GPT.
- Search for youtube videos for each query using [scrapetube](https://github.com/dermasmid/scrapetube).
- Download the videos that were found and subtitles using [yt-dlp](https://github.com/yt-dlp/yt-dlp).
- Detect segments from each video using [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
- Analyze each segment with gpt4o (using images) to filter segments and extract additional useful information (eg overlay text).
- Generate annotations for each segment with GPT using audio transcript (eg instructions) + information extracted from images in the previous step.
- Aggregate segments with annotations into one file
- Cut segments into separate video clips with [ffmpeg](https://ffmpeg.org/).

In the end you'll have a directory with useful video clips and an annotation file, which you can then train a model on.

## 🐬 Installation
- `pip install -r requirements.txt`
- make `.env` file with:
    - `OPENAI_API_KEY` for openai
    - `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` for azure
    - `OPENAI_API_VERSION='2023-07-01-preview'`
- make `config.yaml` file with:
    - `openai.type`: openai/azure
    - `openai.temperature`: the bigger, the more random/creative output will be
    - `openai.deployment`: model for openai / deployment for azure. Needs to be able to do structured output and process images. Tested on gpt4o on azure.
    - `data_dir`: the path where all the results will be saved. Change it for each experiment/dataset.

## 🐙 Usage

Please refer to [example.ipynb](./example.ipynb)

If you have your own videos with descriptions, you can skip the download/filtering steps and move straight to generating annotaions!