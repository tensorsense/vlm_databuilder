import time

def seconds_to_ts(s: float):
    time_str = f'%H:%M:%S.{round((s%1)*1000):03d}'
    return time.strftime(time_str, time.gmtime(s))
