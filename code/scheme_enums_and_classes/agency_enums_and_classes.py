from enum import Enum

class Agency_Predicate():
    def __init__(self):
        self.source = None#tuple()#(source_str, begin,end)

class Agent_of_Predicate():#todo remmember to check agreement in both directions
    def __init__(self):
        self.agent_source = None#  tuple()#(source_str, begin,end)
        self.predicate_source = None#   tuple()#(source_str, begin,end)

class Agency():
    def __init__(self):
        self.source = None#tuple()#(source_str, begin,end)
        self.agency_type = None
        self.agentless_reason = None
        self.position = None
        self.animacy = None
        self.morphology = None

    @staticmethod
    def check_if_sources_are_equal(source_a, source_b):
        if source_a.strip() == source_b.strip():
            return True
        else:
            return False
    @staticmethod
    def check_if_agency_types_are_equal(agancy_type_a, agancy_type_b):
        if agancy_type_a == agancy_type_b:
            return True
        else:
            return False
class Agency_Type(Enum):
    AGENT = "agent"
    EXPERIENCER = "experiencer"
    AGENTLESS = "agent-less"

class Agentless_Reason(Enum):
    EXISENTIAL = "exisential"
    IMPERSONAL_MODAL_VERB = "impersonal modal verb"
    MIDDLE_VERB = "middle_verb"
    INFINITIVAL_SENTENCE = "infinitival sentence"
    NOMINAL_SENTENCE = "nominal sentence"
    PASSIVE_W_O_BY_CLAUSE = "passive w/o by-clause"
    THERE_IS_OR_NOT_INF = "אין/יש (there is/n't) + infintive"
    IMPERATIVE = "Imperative"
    UNSPECIFIED = "unspecified"
    QUESTION = "question"
    NOUN_PHRASE = "noun_phrase"
    OTHER = "other"


class Agency_Position(Enum):
    SUBJECT = "subject"
    INDIRECT_OBJECT = "indirect object"
    EMBEDDED_PRONOUN_SUBJECT = "embedded_pronoun_subject"
    EMBEDDED_PRONOUN_NON_SUBJECT = "embedded_pronoun_non_subject"

class Animacy(Enum):
    HUMAN = "human"
    ANIMATE = "animate"
    INANIMATE = "inanimate"

class Morphology(Enum):
    ONE_PLURAL = "1pl"
    ONE_SINGLE = "1sg"
    TWO_PLURAL = "2pl"
    TWO_SINGLE = "2sg"
    THREE_PLURAL = "3pl"
    THREE_SINGLE = "3sg"
