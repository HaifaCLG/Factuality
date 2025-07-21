import json
import os
import zipfile

from scheme_enums_and_classes import *

parsed_agreements_path = "data\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\parsed"
parsed_rest_path = "data\\processed_knesset\\factuality_manual_annotations\\rest_annotations\\parsed"

avia_parsed_path_agreement = os.path.join(parsed_agreements_path, "avia")
shira_parsed_path_agreement = os.path.join(parsed_agreements_path, "shira")
israel_parsed_path_agreement = os.path.join(parsed_agreements_path, "israel")
avia_parsed_path_rest = os.path.join(parsed_rest_path, "avia")
shira_parsed_path_rest = os.path.join(parsed_rest_path, "shira")
israel_parsed_path_rest = os.path.join(parsed_rest_path, "israel")
parsed_paths =[avia_parsed_path_agreement, shira_parsed_path_agreement, israel_parsed_path_agreement, avia_parsed_path_rest, shira_parsed_path_rest, israel_parsed_path_rest]


def update_feature_dict_values(sent, feature_dict, feature_name):
    if feature_name in sent:
        features = sent[feature_name]
        for feat in features:
            for key in feat.keys():
                if key in feature_dict:
                    feature_dict[key].add(feat[key])



def init_agency_dict():
    agency_dict = {}
    agency_dict['agency_type'] = set()
    agency_dict['agentless_reason'] = set()
    agency_dict['position'] = set()
    agency_dict['animacy'] = set()
    agency_dict['morphology'] = set()
    return agency_dict

def init_check_worthiness_dict():
    check_worthiness_dict = {}
    check_worthiness_dict['check_worthiness_score'] = set()
    check_worthiness_dict['claim_type'] = set()
    check_worthiness_dict['factuality_profile_value'] = set()
    return check_worthiness_dict

def init_esp_dict():
    esp_dict = {}
    esp_dict["esp_type"] = set()
    return esp_dict

def init_stance_dict():
    stance_dict = {}
    stance_dict['confidence_level'] = set()
    stance_dict['stance_type'] = set()
    stance_dict['polarity'] = set()
    stance_dict['reference_type'] =set()
    return stance_dict

def init_quantities_dict():
    quantities_dict = {}
    quantities_dict['quantifier_type'] = set()
    quantities_dict['accuracy'] = set()
    return quantities_dict

def init_NER_dict():
    NER_dict = {}
    NER_dict["entity_type"] = set()
    return NER_dict

def init_anchor_dict():
    anchor_dict = {}
    anchor_dict["this_protocol_anchor"] = set()
    return anchor_dict

def check_agency_values(agency_dict):
    for agency_type in Agency_Type:
        if agency_type.value not in agency_dict['agency_type']:
            print(f"agency_type is not used: {agency_type.value}")
            print(f"used agency types: {agency_dict['agency_type']}")

    for agentless_reason in Agentless_Reason:
        if agentless_reason.value not in agency_dict['agentless_reason']:
            print(f"agentless_reason is not used: {agentless_reason.value}")
            print(f"used agentless_reason: {agency_dict['agentless_reason']}")

    for position in Agency_Position:
        if position.value not in agency_dict['position']:
            print(f"agency position is not used: {position.value}")
            print(f"used positions: {agency_dict['position']}")

    for animacy in Animacy:
        if animacy.value not in agency_dict['animacy']:
            print(f"agency animacy is not used: {animacy.value}")
            print(f"used animacy: {agency_dict['animacy']}")

    for morph in Morphology:
        if morph.value not in agency_dict['morphology']:
            print(f"agency morphology is not used: {morph.value}")
            print(f"used morphologies: {agency_dict['morphology']}")


def check_check_worthiness_values(check_worthiness_dict):
    for score in Check_Worthiness_Score:
        if score.value not in check_worthiness_dict['check_worthiness_score']:
            print(f"check_worthiness_score is not used: {score.value}")
            print(f"used check_worthiness_score: {check_worthiness_dict['check_worthiness_score']}")

    for type in Claim_Type:
        if type.value not in check_worthiness_dict['claim_type']:
            print(f"check_worthiness claim_type is not used: {type.value}")
            print(f"used check_worthiness claim types: {check_worthiness_dict['claim_type']}")

    for profile in Factuality_Profile_Value:
        if profile.value not in check_worthiness_dict['factuality_profile_value']:
            print(f"check_worthiness factuality_profile_value is not used: {profile.value}")
            print(f"used check_worthiness factuality_profile_value: {check_worthiness_dict['factuality_profile_value']}")

def check_esp_values(esp_dict):
    for esp_type in ESP_TYPE:
        if esp_type.value not in esp_dict["esp_type"]:
            print(f"esp_type is not used: {esp_type.value}")
            print(
                f'used esp_types: {esp_dict["esp_type"]}')

def check_stance_values(stance_dict):
    stance_dict['reference_type'] = set()
    for confidence_level in Stance_Confidence_Level:
        if confidence_level.value not in stance_dict["confidence_level"]:
            print(f"stance confidence_level is not used: {confidence_level.value}")
            print(
                f'used confidence_levels: {stance_dict["confidence_level"]}')
    for stance_type in Stance_Type:
        if stance_type.value not in stance_dict["stance_type"]:
            print(f"stance type is not used: {stance_type.value}")
            print(
                f'used stance types: {stance_dict["stance_type"]}')
    for polarity in Stance_Polarity:
        if polarity.value not in stance_dict["polarity"]:
            print(f"stance polarity is not used: {polarity.value}")
            print(
                f'used polarity: {stance_dict["polarity"]}')

    for ref in Stance_Reference_Type:
        if ref.value not in stance_dict["reference_type"]:
            print(f"stance refernce type is not used: {ref.value}")
            print(
                f'used refernce types: {stance_dict["reference_type"]}')
def check_quantities_values(quantities_dict):
    for quantifier_type in Quantifier_Type:
        if quantifier_type.value not in quantities_dict["quantifier_type"]:
            print(f"Quantifier_Type is not used: {quantifier_type.value}")
            print(
                f'used Quantifier_Types: {quantities_dict["quantifier_type"]}')
    for accuracy in QUANTIFIER_ACCURACY:
        if accuracy.value not in quantities_dict["accuracy"]:
            print(f"QUANTIFIER_ACCURACY is not used: {accuracy.value}")
            print(
                f'used QUANTIFIER_ACCURACies: {quantities_dict["accuracy"]}')
def check_ner_values(NER_dict):
    for ner in ENTITY_TYPE:
        if ner.value not in NER_dict["entity_type"]:
            print(f"entity_type is not used: {ner.value}")
            print(
                f'used entity_types: {NER_dict["entity_type"]}')


def check_anchor_values(anchor_dict):
    if anchor_dict['this_protocol_anchor']:
        print("anchor was used")

if __name__ == '__main__':
    agency_dict = init_agency_dict()
    check_worthiness_dict = init_check_worthiness_dict()
    esp_dict = init_esp_dict()
    stance_dict = init_stance_dict()
    quantities_dict = init_quantities_dict()
    NER_dict = init_NER_dict()
    anchor_dict = init_anchor_dict()

    for parsed_dir in parsed_paths:
        for sent in os.listdir(parsed_dir):
            sent_path = os.path.join(parsed_dir, sent)
            try:
                with open(sent_path, encoding="utf-8") as file:
                    sent_json = json.load(file)
            except Exception as e:
                print(f'couldnt parse json: {sent_path}')
                continue
            update_feature_dict_values(sent_json, agency_dict, "agencies")
            update_feature_dict_values(sent_json, check_worthiness_dict, "check_worthiness")
            update_feature_dict_values(sent_json, esp_dict, "esp_profiles")
            update_feature_dict_values(sent_json, stance_dict, "stances")
            update_feature_dict_values(sent_json, quantities_dict, "quantities")
            update_feature_dict_values(sent_json, NER_dict, "named_entites")
            update_feature_dict_values(sent_json, anchor_dict, "this_protocol_anchors")


    check_agency_values(agency_dict)
    check_check_worthiness_values(check_worthiness_dict)
    check_esp_values(esp_dict)
    check_stance_values(stance_dict)
    check_quantities_values(quantities_dict)
    check_ner_values(NER_dict)
    check_anchor_values(anchor_dict)






