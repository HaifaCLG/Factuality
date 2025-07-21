from enum import Enum


class Check_Worthiness():
    def __init__(self):
        # self.start_claim_token =None#tuple()  # (source_str, begin, end)
        self.check_worthiness_score = None
        self.claim_type = None
        self.factuality_profile_source = None#string of name
        self.factuality_profile_value = None


class Check_Worthiness_Score(Enum):
    WORTH_CHECKING = "worth checking"
    NOT_WORTH_CHECKING = "not worth checking"
    NOT_A_FACTUAL_PROPOSITION = "not a factual proposition"

class Claim_Type(Enum):
    PERSONAL_EXP = "personal experience"
    QUANTITY = "quantity in the past or present"
    CORR_OR_CAUS = "correlation or causation"
    LAWS_OR_RULES = "current laws or rules of operation"
    PREDICTION = "prediction"
    OTHER = "other type of claim"
    NOT_A_CLAIM = "not a claim"

class Factuality_Profile_Value(Enum):
    CT_PLUS = "CT+"
    PS_PLUS = "PS+"
    PR_PLUS = "PR+"
    CT_MINUS = "CT-"
    PR_MINUS = "PR-"
    PS_MINUS = "PS-"
    CT_U = "CTu"
    U_u = "Uu"

class Factuality_Profile_Source_Name(Enum):
    SPEAKER_FROM_META_DATA_SOURCE = "speaker_from_metadata"
