from enum import Enum
class Quantity():
    def __init__(self):
        self.expression = None #(source_str, begin,end)
        self.quantifier_type = None
        self.accuracy = None
        self.the_whole = None#(source_str, begin,end)



class Quantifier_Type(Enum):
    EXISENTIAL = "exisential"
    PARTIAL = "partial"
    UNIVERSAL = "universal"
    IRRELEVANT = "irrelevant"
    QUANTITY_EXP = "quantity_expression"

class QUANTIFIER_ACCURACY(Enum):
    ACCURATE = "accurate"
    ESTIMATE = "estimate"
    OBSCURE = "obscure"