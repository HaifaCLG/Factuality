from enum import Enum
class Stance():
    def __init__(self):
        self.confidence_level = None
        self.stance_type = None
        self.polarity = None
        self.reference_name = None#tuple()#(source_str, begin,end)
        self.reference_type = None

class Stance_polarity_indication():
    def __init__(self):
        self.source = None#tuple()#(source_str, begin,end)
class Stance_Confidence_Level(Enum):
    HIGH = "high"
    MIDDLE = "mid"
    LOW = "low"
    IRRELEVANT = "irrelevant"

class Stance_Type(Enum):#NOT RELEVANT=None
    EPISTEMIC = "epistemic"
    EFFECTIVE = "effective"

class Stance_Polarity(Enum):
    NEGATIVE = "negative"
    POSITIVE = "positive"
    UNDERSPECIFIED = "underspecified"

class Stance_Reference_Type(Enum):
    ARTICLE = "article"
    BOOK = "book"
    LAWS = "laws"
    NUMBERS = "numbers"
    QUOTING = "quoting an expert or authority figure"
    RESEARCH = "research"
    STATS = "stats"
    SURVEY = "survey"
    OTHER = "other"