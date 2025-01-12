from pathlib import Path
from inspect import currentframe, getframeinfo


# BASE_DIR = "/Users/martin.ong/Desktop/grammar-inference"
BASE_DIR = "/home/martin/Desktop/uni/2024/grammar-inference"


def to_rel(full_path):
    return Path(full_path).relative_to(BASE_DIR)


def special_print(func):
    def wrapped_func(*args, **kwargs):
        if curr_frame := currentframe():
            if prev_frame := curr_frame.f_back:
                frameinfo = getframeinfo(prev_frame)
                return func(f"{to_rel(frameinfo.filename)} {frameinfo.lineno}:", *args, **kwargs)
        return func(args, kwargs)
    return wrapped_func


print = special_print(print)
