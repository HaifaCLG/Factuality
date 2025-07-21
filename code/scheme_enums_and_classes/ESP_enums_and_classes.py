from enum import Enum

class ESP_Profile():
    def __init__(self):
        self.esp = None#tuple()(source_str, begin,end)
        self.esp_type = None

class ESP_TYPE(Enum):
   SIP = "SIP"
   NSIP = "NSIP"
