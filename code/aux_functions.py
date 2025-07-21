from config import *


def check_if_lists_are_identical(list_a,list_b):
    if len(list_a) != len(list_b):
        return False
    else:
        if list_a == list_b:
            return True
        else:
            return False
