from pathlib import Path
from inspect import currentframe, getframeinfo, signature
import sys


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
# print = print


def is_user_defined(frame):
    """Check if the function is inside the user's project directory."""
    filename = frame.f_code.co_filename
    return filename.startswith(BASE_DIR)  # Only track functions within your codebase

def is_dunder_method(func_name):
    """Check if a function is a dunder method (e.g., __init__, __call__)."""
    return func_name.startswith("__") and func_name.endswith("__")

def format_args(frame):
    """Extract and format function arguments."""
    # TODO: this doesn't work properly - although locals_ definitely contains the arguments
    func_name = frame.f_code.co_name
    locals_ = frame.f_locals  # Get local variables (function arguments)
    
    # Try to get argument names using signature
    try:
        func_sig = signature(frame.f_globals.get(func_name, lambda: None))
        param_names = list(func_sig.parameters.keys())
        args_repr = ", ".join(f"{name}={locals_.get(name, '?')}" for name in param_names)
    except ValueError:
        args_repr = ", ".join(f"{k}={v!r}" for k, v in locals_.items())  # Backup method

    return f"({args_repr})"

def trace_calls(frame, event, arg):
    """Global function call tracer for user-defined, non-dunder functions with arguments."""
    if event == "call" and is_user_defined(frame):
        function_name = frame.f_code.co_name
        if not is_dunder_method(function_name):  # Ignore dunder methods
            frameinfo = getframeinfo(frame)
            args_repr = format_args(frame)  # Get function arguments
            print(f"Function {function_name}{args_repr} called from {frameinfo.filename}:{frameinfo.lineno}")
    return trace_calls

# Enable tracing
# sys.setprofile(trace_calls)
