from enum import Enum
class NER():
    def __init__(self):
        self.entity_name = None#tuple()#(source_str, begin,end)
        self.entity_type = None

class ENTITY_TYPE(Enum):
    PER = "PER"
    ORG = "ORG"
    LOC = "LOC"
    GPE = "GPE"
    FAC = "FAC"
    EVE = "EVE"
    DUC = "DUC"
    WOA = "WOA"
    ANG = "ANG"
    TTL = "TTL"
    TIMEX = "TIMEX"
    MISC = "MISC"