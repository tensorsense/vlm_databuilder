import yt_dlp

from .core.config import DatagenConfig


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
    YDL_OPTIONS = {**YDL_OPTIONS_DEFAULT}
    YDL_OPTIONS['outtmpl'] = {'default': config.video_dir.as_posix() + '/%(id)s.%(ext)s', 'subtitle': config.sub_dir.as_posix() + '/%(id)s'}
    if yt_dlp_opts:
        # override options if necessary, eg languages, formats, etc
        # refer to yt_dlp.YoutubeDL class
        YDL_OPTIONS = {**YDL_OPTIONS, **yt_dlp_opts}
    with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={i}' for i in ids])
