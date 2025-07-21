from enum import Enum

class Time_EXP_Range():
    def __init__(self):
        self.original_time_tokens = None#tuple(source_str, begin,end)
        self.formatted_begin_date = None
        self.formatted_end_date = None