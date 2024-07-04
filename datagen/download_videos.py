import yt_dlp

from .core.config import DatagenConfig
from .core.sub_utils import vtt_to_txt


# yt-dlp --write-auto-subs --sub-langs "en" --sub-format vtt  --no-overwrites -o 'videos/%(id)s.%(ext)s' -o 'subtitle:subs/%(id)s' -f mp4  https://www.youtube.com/watch?v=Me1jEspRQEg

# 'paths': {'default': '../tmp/', 'subtitle': '../tmp/subs'},
YDL_OPTIONS_DEFAULT = {
    # "quiet":    True,
    # "simulate": True,
    # "forceurl": True,
    # 'verbose': True,
    'writeautomaticsub': True,
    # 'writesubtitles': True,
    'subtitleslangs': ['en'],
    'subtitlesformat': 'vtt',
    'overwrites': False,
    'format': 'mp4',
}

def download_videos(ids, config: DatagenConfig, yt_dlp_opts={}):
    # do not download ids that have videos downloaded
    #TODO force download subs
    ids = set(ids) - set([v.stem for v in config.get_videos()])
    YDL_OPTIONS = {**YDL_OPTIONS_DEFAULT}
    YDL_OPTIONS['outtmpl'] = {'default': config.video_dir.as_posix() + '/%(id)s.%(ext)s', 'subtitle': config.sub_dir.as_posix() + '/%(id)s'}
    if yt_dlp_opts:
        # override options if necessary, eg languages, formats, etc
        # refer to yt_dlp.YoutubeDL class
        YDL_OPTIONS = {**YDL_OPTIONS, **yt_dlp_opts}
    with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
        # TODO: download each video separately in tqdm loop with quiet yt-dlp?
        ydl.download([f'https://www.youtube.com/watch?v={i}' for i in ids])

    for sub_path in config.get_subs():
        if '.' in sub_path.stem:
            sub_path.rename(sub_path.with_stem(sub_path.stem.split('.')[0]))

    for sub_path in config.get_subs():
        transcript_path = config.transcript_dir / sub_path.with_suffix('.txt').name
        if transcript_path.exists():
            continue
        with open(transcript_path, 'w') as f:
            f.write(vtt_to_txt(sub_path))

