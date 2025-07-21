import itertools
import json
import os
import ast

from datetime import datetime

import numpy as np

from config import *
from scheme_enums_and_classes import *
from aux_functions import *
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ALL_FEATURES_AGREEMENT_IN_ONE_FILE = True
ONLY_SAME_NUM_OF_CLAIMS =True
IGNORE_NOT_A_FACTUAL_PROPOSITION = False
ONLY_ONE_CLAIM = True
ONLY_SAME_CLAIMS = True
IGNORE_NOT_A_CLAIM = False

def remove_files_not_in_list(dir, files_to_keep):
    files = os.listdir(dir)
    for file in files:
        if file not in files_to_keep:
            file_path = os.path.join(dir, file)
            os.remove(file_path)

def keep_only_same_files_in_each_annotator_dir(path_a, path_b, path_c):
    a_files = os.listdir(path_a)
    b_files = os.listdir(path_b)
    c_files = os.listdir(path_c)
    a_b_common_list = set(a_files).intersection(b_files)
    common_list = set(a_b_common_list).intersection(c_files)
    remove_files_not_in_list(path_a, common_list)
    remove_files_not_in_list(path_b, common_list)
    remove_files_not_in_list(path_c, common_list)




def check_sent_feature_list_agreement(list_a,list_b, list_c, sent_text, sent_name="", counter=None ):
    #a=shira, b= avia, c= israel

    sentence_to_analayze_dict = {}
    sentence_to_analayze_dict["sent_name"] = sent_name
    sentence_to_analayze_dict["sent_text"] = sent_text
    sentence_to_analayze_dict["Shira_annotations"] = list_a
    sentence_to_analayze_dict["Avia_annotations"] = list_b
    sentence_to_analayze_dict["Israel_annotations"] = list_c
    agreement = 3
    res_a_b = check_if_lists_are_identical(list_a, list_b)
    res_a_c = check_if_lists_are_identical(list_a, list_c)
    if res_a_b and res_a_c:
        agreement = 3
    elif res_a_b and not res_a_c:
        agreement = 2
        if len(list_a) == len(list_b) and len(list_a)==len(list_c):
            for elem_a, elem_c in zip(list_a, list_c):
                if elem_a != elem_c:
                    statement = f'two agreed on: {elem_a}, and one said: {elem_c}'
                    if counter:
                        count = counter.get(statement, 0)
                        count +=1
                        counter[statement] = count

    elif res_a_c and not res_a_b:
        agreement = 2
        if len(list_a) == len(list_b) and len(list_a)==len(list_c):
            for elem_a, elem_b in zip(list_a, list_b):
                if elem_a != elem_b:
                    statement =f'two agreed on: {elem_a}, and one said: {elem_b}'
                    # if elem_a == "not worth checking" and elem_b == "worth checking" or elem_a=="worth checking" and elem_b=="not worth checking"
                    if counter:
                        count = counter.get(statement, 0)
                        count += 1
                        counter[statement] = count


    elif res_a_b == False and res_a_c == False:
        if check_if_lists_are_identical(list_b, list_c):
            if len(list_a) == len(list_b) and len(list_a) == len(list_c):
                for elem_b, elem_a in zip(list_b, list_a):
                    if elem_b != elem_a:
                        statement = f'two agreed on: {elem_b}, and one said: {elem_a}'
                        if counter:
                            count = counter.get(statement, 0)
                            count += 1
                            counter[statement] = count
            agreement = 2


        else:
            agreement = 1
            # print(f"agreement 1 sent: {sent_text}")


            if len(list_a) == len(list_b) and len(list_a) == len(list_c):
                for elem_a,elem_b, elem_c in zip(list_a, list_b, list_c):
                    if elem_a != elem_b and elem_b != elem_c:
                        statement = f'no agreement about: {elem_a} or: {elem_b} or: {elem_c}'
                        if counter:
                            count = counter.get(statement, 0)
                            count += 1
                            counter[statement] = count
    return agreement, sentence_to_analayze_dict


def get_annotators_sorted_sentences_names(annotator_1_sentences_annotations_path,annotator_2_sentences_annotations_path , annotator_3_sentences_annotations_path):
    annotator_1_sentences = os.listdir(annotator_1_sentences_annotations_path)
    annotator_1_sentences.sort()
    annotator_2_sentences = os.listdir(annotator_2_sentences_annotations_path)
    annotator_2_sentences.sort()
    annotator_3_sentences = os.listdir(annotator_3_sentences_annotations_path)
    annotator_3_sentences.sort()
    return annotator_1_sentences, annotator_2_sentences, annotator_3_sentences


def get_sent_annotators_annotations(sents_path_1, sents_path_2, sents_path_3, sent_name):
    sent_annotated_1_path = os.path.join(sents_path_1, sent_name)
    sent_annotated_2_path = os.path.join(sents_path_2, sent_name)
    sent_annotated_3_path = os.path.join(sents_path_3, sent_name)
    with open(sent_annotated_1_path, encoding='utf-8') as file:
        sent_annotations_1 = json.load(file)
    with open(sent_annotated_2_path, encoding='utf-8') as file:
        sent_annotations_2 = json.load(file)
    with open(sent_annotated_3_path, encoding='utf-8') as file:
        sent_annotations_3 = json.load(file)
    return sent_annotations_1, sent_annotations_2, sent_annotations_3

def get_sent_all_time_expressions_features_agreement(sent_annotations_1,sent_annotations_2, sent_annotations_3, sent_name):
    # time_exps_1 = sent_annotations_1['time_expression']
    # if time_exps_1:
    #     original_time_tokens_1, formatted_timeEXP_1 = get_all_lists_of_time_exp_features_in_sent(time_exps_1)
    # else:
    #     original_time_tokens_1 = []
    #     formatted_timeEXP_1 = []

    time_range_exp_1 = sent_annotations_1['time_expression_ranges']
    if time_range_exp_1:
        original_range_time_tokens_1, formatted_begin_date_1, formatted_end_date_1 = get_all_lists_of_time_range_exp_features_in_sent(time_range_exp_1)
    else:
        original_range_time_tokens_1 = []
        formatted_begin_date_1 = []
        formatted_end_date_1 = []

    # time_exps_2 = sent_annotations_2['time_expression']
    # if time_exps_2:
    #     original_time_tokens_2, formatted_timeEXP_2 = get_all_lists_of_time_exp_features_in_sent(time_exps_2)
    # else:
    #     original_time_tokens_2 = []
    #     formatted_timeEXP_2 = []

    time_range_exp_2 = sent_annotations_2['time_expression_ranges']
    if time_range_exp_2:
        original_range_time_tokens_2, formatted_begin_date_2, formatted_end_date_2 = get_all_lists_of_time_range_exp_features_in_sent(
            time_range_exp_2)
    else:
        original_range_time_tokens_2 = []
        formatted_begin_date_2 = []
        formatted_end_date_2 = []

    # time_exps_3 = sent_annotations_3['time_expression']
    # if time_exps_3:
    #     original_time_tokens_3, formatted_timeEXP_3 = get_all_lists_of_time_exp_features_in_sent(time_exps_3)
    # else:
    #     original_time_tokens_3 = []
    #     formatted_timeEXP_3 = []

    time_range_exp_3 = sent_annotations_3['time_expression_ranges']
    if time_range_exp_3:
        original_range_time_tokens_3, formatted_begin_date_3, formatted_end_date_3 = get_all_lists_of_time_range_exp_features_in_sent(
            time_range_exp_3)
    else:
        original_range_time_tokens_3 = []
        formatted_begin_date_3 = []
        formatted_end_date_3 = []
#TODO this is before merged time exp and time range
    # all_time_expressions_features_list_1 = list(
    #     itertools.chain(original_time_tokens_1, formatted_timeEXP_1, original_range_time_tokens_1, formatted_begin_date_1, formatted_end_date_1))
    # all_time_expressions_features_list_2 = list(
    #     itertools.chain(original_time_tokens_2, formatted_timeEXP_2, original_range_time_tokens_2,
    #                     formatted_begin_date_2, formatted_end_date_2))
    # all_time_expressions_features_list_3 = list(
    #     itertools.chain(original_time_tokens_3, formatted_timeEXP_3, original_range_time_tokens_3,
    #                     formatted_begin_date_3, formatted_end_date_3))
    all_time_expressions_features_list_1 = list(
        itertools.chain(original_range_time_tokens_1,
                        formatted_begin_date_1, formatted_end_date_1))
    all_time_expressions_features_list_2 = list(
        itertools.chain(original_range_time_tokens_2,
                        formatted_begin_date_2, formatted_end_date_2))
    all_time_expressions_features_list_3 = list(
        itertools.chain( original_range_time_tokens_3,
                        formatted_begin_date_3, formatted_end_date_3))


    all_time_expressions_features_sent_agreement, all_time_expressions_features_agreement_record = check_sent_feature_list_agreement(
        all_time_expressions_features_list_1, all_time_expressions_features_list_2, all_time_expressions_features_list_3,
        sent_annotations_1["sent_text"],
        sent_name=sent_name)

    all_time_expressions_features_agreement_record.pop("Shira_annotations")
    all_time_expressions_features_agreement_record.pop("Avia_annotations")
    all_time_expressions_features_agreement_record.pop("Israel_annotations")


    all_time_expressions_features_agreement_record["Shira_original_range_time_tokens"] = original_range_time_tokens_1
    all_time_expressions_features_agreement_record["Avia_original_range_time_tokens"] = original_range_time_tokens_2
    all_time_expressions_features_agreement_record["Israel_original_range_time_tokens"] = original_range_time_tokens_3

    begin_years_1, begin_months_1, begin_days_1, begin_hours_1, begin_minutes_1 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_begin_date_1)
    begin_years_2, begin_months_2, begin_days_2, begin_hours_2, begin_minutes_2 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_begin_date_2)
    begin_years_3, begin_months_3, begin_days_3, begin_hours_3, begin_minutes_3 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_begin_date_3)

    all_time_expressions_features_agreement_record["Shira_begin_years"] = begin_years_1
    all_time_expressions_features_agreement_record["Avia_begin_years"] = begin_years_2
    all_time_expressions_features_agreement_record["Israel_begin_years"] = begin_years_3

    all_time_expressions_features_agreement_record["Shira_begin_months"] = begin_months_1
    all_time_expressions_features_agreement_record["Avia_begin_months"] = begin_months_2
    all_time_expressions_features_agreement_record["Israel_begin_months"] = begin_months_3

    all_time_expressions_features_agreement_record["Shira_begin_days"] = begin_days_1
    all_time_expressions_features_agreement_record["Avia_begin_days"] = begin_days_2
    all_time_expressions_features_agreement_record["Israel_begin_days"] = begin_days_3

    all_time_expressions_features_agreement_record["Shira_begin_hours"] = begin_hours_1
    all_time_expressions_features_agreement_record["Avia_begin_hours"] = begin_hours_2
    all_time_expressions_features_agreement_record["Israel_begin_hours"] = begin_hours_3

    all_time_expressions_features_agreement_record["Shira_begin_minutes"] = begin_minutes_1
    all_time_expressions_features_agreement_record["Avia_begin_minutes"] = begin_minutes_1
    all_time_expressions_features_agreement_record["Israel_begin_minutes"] = begin_minutes_1


    end_years_1, end_months_1, end_days_1, end_hours_1, end_minutes_1 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_end_date_1)
    end_years_2, end_months_2, end_days_2, end_hours_2, end_minutes_2 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_end_date_2)
    end_years_3, end_months_3, end_days_3, end_hours_3, end_minutes_3 = get_year_month_day_hour_minute_lists_from_formatted_dates_list(
        formatted_end_date_3)

    all_time_expressions_features_agreement_record["Shira_end_years"] = end_years_1
    all_time_expressions_features_agreement_record["Avia_end_years"] = end_years_2
    all_time_expressions_features_agreement_record["Israel_end_years"] = end_years_3

    all_time_expressions_features_agreement_record["Shira_end_months"] = end_months_1
    all_time_expressions_features_agreement_record["Avia_end_months"] = end_months_2
    all_time_expressions_features_agreement_record["Israel_end_months"] = end_months_3

    all_time_expressions_features_agreement_record["Shira_end_days"] = end_days_1
    all_time_expressions_features_agreement_record["Avia_end_days"] = end_days_2
    all_time_expressions_features_agreement_record["Israel_end_days"] = end_days_3

    all_time_expressions_features_agreement_record["Shira_end_hours"] = end_hours_1
    all_time_expressions_features_agreement_record["Avia_end_hours"] = end_hours_2
    all_time_expressions_features_agreement_record["Israel_end_hours"] = end_hours_3

    all_time_expressions_features_agreement_record["Shira_end_minutes"] = end_minutes_1
    all_time_expressions_features_agreement_record["Avia_end_minutes"] = end_minutes_2
    all_time_expressions_features_agreement_record["Israel_end_minutes"] = end_minutes_3
    if ALL_FEATURES_AGREEMENT_IN_ONE_FILE:
        return all_time_expressions_features_list_1, all_time_expressions_features_list_2, all_time_expressions_features_list_3, all_time_expressions_features_agreement_record
    else:
        return all_time_expressions_features_sent_agreement, all_time_expressions_features_agreement_record


def get_year_month_day_hour_minute_lists_from_formatted_dates_list(formatted_timeEXP):
    DATE_FORMAT = '%Y-%m-%d %H:%M'
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    for date_str in formatted_timeEXP:
        if date_str:
            year = date_str.split("-")[0]
            years.append(year)
            month = date_str.split("-")[1]
            months.append(month)
            day = date_str.split("-")[2].split()[0]
            days.append(day)
            hour = date_str.split()[-1].split(":")[0]
            hours.append(hour)
            minute = date_str.split()[-1].split(":")[1]
            minutes.append(minute)
        else:
            years.append('None')
            months.append('None')
            days.append('None')
            hours.append('None')
            minutes.append('None')
    return years, months, days, hours, minutes


def get_sent_all_quantities_features_agreement(sent_annotations_1,sent_annotations_2,sent_annotations_3, sent_name):
    quantities_1 = sent_annotations_1['quantities']
    if quantities_1:
        expressions_1, quantifiers_types_1, accuracy_1, the_whole_1 = get_all_lists_of_quantities_features_in_sent(quantities_1)
    else:
        expressions_1 = []
        quantifiers_types_1 = []
        accuracy_1 = []
        the_whole_1 = []

    quantities_2 = sent_annotations_2['quantities']
    if quantities_2:
        expressions_2, quantifiers_types_2, accuracy_2, the_whole_2 = get_all_lists_of_quantities_features_in_sent(quantities_1)
    else:
        expressions_2 = []
        quantifiers_types_2 = []
        accuracy_2 = []
        the_whole_2 = []

    quantities_3 = sent_annotations_3['quantities']
    if quantities_3:
        expressions_3, quantifiers_types_3, accuracy_3, the_whole_3 = get_all_lists_of_quantities_features_in_sent(
            quantities_3)
    else:
        expressions_3 = []
        quantifiers_types_3 = []
        accuracy_3 = []
        the_whole_3 = []

    all_quantities_features_list_1 = list(
        itertools.chain(expressions_1, quantifiers_types_1, accuracy_1, the_whole_1))
    all_quantities_features_list_2 = list(
        itertools.chain(expressions_2, quantifiers_types_2, accuracy_2, the_whole_2))
    all_quantities_features_list_3 = list(
        itertools.chain(expressions_3, quantifiers_types_3, accuracy_3, the_whole_3))

    all_quantities_features_sent_agreement, all_quantities_features_agreement_record = check_sent_feature_list_agreement(all_quantities_features_list_1, all_quantities_features_list_2, all_quantities_features_list_3, sent_annotations_1["sent_text"],
     sent_name=sent_name)

    all_quantities_features_agreement_record.pop("Shira_annotations")
    all_quantities_features_agreement_record.pop("Avia_annotations")
    all_quantities_features_agreement_record.pop("Israel_annotations")

    all_quantities_features_agreement_record["Shira_expressions"] = expressions_1
    all_quantities_features_agreement_record["Avia_expressions"] = expressions_2
    all_quantities_features_agreement_record["Israel_expressions"] = expressions_3

    all_quantities_features_agreement_record["Shira_quantifiers_types"] = quantifiers_types_1
    all_quantities_features_agreement_record["Avia_quantifiers_types"] = quantifiers_types_2
    all_quantities_features_agreement_record["Israel_quantifiers_types"] = quantifiers_types_3

    all_quantities_features_agreement_record["Shira_accuracy"] = accuracy_1
    all_quantities_features_agreement_record["Avia_accuracy"] = accuracy_2
    all_quantities_features_agreement_record["Israel_accuracy"] = accuracy_3

    all_quantities_features_agreement_record["Shira_the_whole"] = the_whole_1
    all_quantities_features_agreement_record["Avia_the_whole"] = the_whole_2
    all_quantities_features_agreement_record["Israel_the_whole"] = the_whole_3
    if ALL_FEATURES_AGREEMENT_IN_ONE_FILE:
        return all_quantities_features_list_1, all_quantities_features_list_2, all_quantities_features_list_3, all_quantities_features_agreement_record
    else:
        return all_quantities_features_sent_agreement, all_quantities_features_agreement_record


def get_sent_all_agency_and_esp_features_agreement(sent_annotations_1,sent_annotations_2,sent_annotations_3,sent_name):
    agencies_1 = sent_annotations_1['agencies']
    agency_sources_1, agency_types_1, agentless_reason_1, position_1, animacy_1, morphologies_1 = get_all_lists_of_agency_features_in_sent(agencies_1)
    agency_predicates_1 = sent_annotations_1['agency_predicates']
    if agency_predicates_1:
        predicate_source_1 = get_all_lists_of_agency_predicate_features_in_sent(agency_predicates_1)
    else:
        predicate_source_1 = []
    agents_of_predicates_relation_1 = sent_annotations_1["agents_of_predicates"]
    if agents_of_predicates_relation_1:
        agent_source_predicate_source_relations_list_1 =  get_all_lists_of_agents_of_predicates_relation_features_in_sent(agents_of_predicates_relation_1)
    else:
        agent_source_predicate_source_relations_list_1 = []
    esp_profiles_1 = sent_annotations_1['esp_profiles']
    if esp_profiles_1:
        esps_1 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_1, 'esp')
        esps_types_1 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_1, 'esp_type')
    else:
        esps_1 = []
        esps_types_1 = []
    agencies_2 = sent_annotations_2['agencies']
    agency_sources_2, agency_types_2, agentless_reason_2, position_2, animacy_2, morphologies_2 = get_all_lists_of_agency_features_in_sent(agencies_2)
    agency_predicates_2 = sent_annotations_2['agency_predicates']
    if agency_predicates_2:
        predicate_source_2 = get_all_lists_of_agency_predicate_features_in_sent(agency_predicates_2)
    else:
        predicate_source_2 = []
    agents_of_predicates_relation_2 = sent_annotations_2["agents_of_predicates"]
    if agents_of_predicates_relation_2:
        agent_source_predicate_source_relations_list_2 = get_all_lists_of_agents_of_predicates_relation_features_in_sent(
            agents_of_predicates_relation_2)
    else:
        agent_source_predicate_source_relations_list_2 = []
    esp_profiles_2 = sent_annotations_2['esp_profiles']
    if esp_profiles_2:
        esps_2 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_2, 'esp')
        esps_types_2 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_2, 'esp_type')
    else:
        esps_2 = []
        esps_types_2 = []
    agencies_3 = sent_annotations_3['agencies']
    agency_sources_3, agency_types_3, agentless_reason_3, position_3, animacy_3, morphologies_3 = get_all_lists_of_agency_features_in_sent(agencies_3)
    agency_predicates_3 = sent_annotations_3['agency_predicates']
    if agency_predicates_3:
        predicate_source_3 = get_all_lists_of_agency_predicate_features_in_sent(agency_predicates_3)
    else:
        predicate_source_3 = []
    agents_of_predicates_relation_3 = sent_annotations_3["agents_of_predicates"]
    if agents_of_predicates_relation_3:
        agent_source_predicate_source_relations_list_3  = get_all_lists_of_agents_of_predicates_relation_features_in_sent(
            agents_of_predicates_relation_3)
    else:
        agent_source_predicate_source_relations_list_3 = []
    esp_profiles_3 = sent_annotations_3['esp_profiles']
    if esp_profiles_3:
        esps_3 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_3, 'esp')
        esps_types_3 = get_list_of_all_values_of_certain_layer_feature_in_sent(esp_profiles_3, 'esp_type')
    else:
        esps_3 = []
        esps_types_3 = []
    all_agency_and_esp_features_list_1 = list(itertools.chain(agency_sources_1, agency_types_1, agentless_reason_1, position_1 , animacy_1, morphologies_1, esps_1,predicate_source_1, agent_source_predicate_source_relations_list_1,esps_types_1))
    all_agency_and_esp_features_list_2 = list(itertools.chain(agency_sources_2, agency_types_2, agentless_reason_2, position_2 , animacy_2, morphologies_2, esps_2, predicate_source_2, agent_source_predicate_source_relations_list_2,esps_types_2))
    all_agency_and_esp_features_list_3 = list(itertools.chain(agency_sources_3, agency_types_3, agentless_reason_3, position_3 , animacy_3, morphologies_3, esps_3, predicate_source_3, agent_source_predicate_source_relations_list_3,esps_types_3))

    all_agency_and_esp_features_sent_agreement, all_agency_and_esp_features_agreement_record = check_sent_feature_list_agreement(
        all_agency_and_esp_features_list_1,
        all_agency_and_esp_features_list_2,
        all_agency_and_esp_features_list_3,
        sent_annotations_1["sent_text"],
        sent_name=sent_name)

    all_agency_and_esp_features_agreement_record.pop("Shira_annotations")
    all_agency_and_esp_features_agreement_record.pop("Avia_annotations")
    all_agency_and_esp_features_agreement_record.pop("Israel_annotations")

    all_agency_and_esp_features_agreement_record["Shira_agency_sources"] = agency_sources_1
    all_agency_and_esp_features_agreement_record["Avia_agency_sources"] = agency_sources_2
    all_agency_and_esp_features_agreement_record["Israel_agency_sources"] = agency_sources_3

    all_agency_and_esp_features_agreement_record["Shira_agency_types"] = agency_types_1
    all_agency_and_esp_features_agreement_record["Avia_agency_types"] = agency_types_2
    all_agency_and_esp_features_agreement_record["Israel_agency_types"] = agency_types_3

    all_agency_and_esp_features_agreement_record["Shira_agentless_reason"] = agentless_reason_1
    all_agency_and_esp_features_agreement_record["Avia_agentless_reason"] = agentless_reason_2
    all_agency_and_esp_features_agreement_record["Israel_agentless_reason"] = agentless_reason_3

    all_agency_and_esp_features_agreement_record["Shira_position"] = position_1
    all_agency_and_esp_features_agreement_record["Avia_position"] = position_2
    all_agency_and_esp_features_agreement_record["Israel_position"] = position_3

    all_agency_and_esp_features_agreement_record["Shira_animacy"] = animacy_1
    all_agency_and_esp_features_agreement_record["Avia_animacy"] = animacy_2
    all_agency_and_esp_features_agreement_record["Israel_animacy"] = animacy_3

    all_agency_and_esp_features_agreement_record["Shira_morphology"] = morphologies_1
    all_agency_and_esp_features_agreement_record["Avia_morphology"] = morphologies_2
    all_agency_and_esp_features_agreement_record["Israel_morphology"] = morphologies_3

    all_agency_and_esp_features_agreement_record["Shira_agency_predicates"] = agency_predicates_1
    all_agency_and_esp_features_agreement_record["Avia_agency_predicates"] = agency_predicates_2
    all_agency_and_esp_features_agreement_record["Israel_agency_predicates"] = agency_predicates_3

    all_agency_and_esp_features_agreement_record["Shira_agents_of_predicates_relation"] = agent_source_predicate_source_relations_list_1
    all_agency_and_esp_features_agreement_record["Avia_agents_of_predicates_relation"] = agent_source_predicate_source_relations_list_2
    all_agency_and_esp_features_agreement_record["Israel_agents_of_predicates_relation"] = agent_source_predicate_source_relations_list_3

    all_agency_and_esp_features_agreement_record["Shira_esp"] = esps_1
    all_agency_and_esp_features_agreement_record["Avia_esp"] = esps_2
    all_agency_and_esp_features_agreement_record["Israel_esp"] = esps_3

    all_agency_and_esp_features_agreement_record["Shira_esp_type"] = esps_types_1
    all_agency_and_esp_features_agreement_record["Avia_esp_type"] = esps_types_2
    all_agency_and_esp_features_agreement_record["Israel_esp_type"] = esps_types_3

    if ALL_FEATURES_AGREEMENT_IN_ONE_FILE:
        return all_agency_and_esp_features_list_1, all_agency_and_esp_features_list_2, all_agency_and_esp_features_list_3, all_agency_and_esp_features_agreement_record
    else:
        return all_agency_and_esp_features_sent_agreement, all_agency_and_esp_features_agreement_record


def get_sent_all_stance_features_agreement(sent_annotations_1, sent_annotations_2, sent_annotations_3, sent_name):
    keep_stance_type = False
    stances_1 = sent_annotations_1['stances']
    confidence_levels_1, stance_types_1, polarities_1, reference_names_1, reference_types_1 = get_all_lists_of_stance_features_in_sent(stances_1)
    stances_2 = sent_annotations_2['stances']
    confidence_levels_2, stance_types_2, polarities_2, reference_names_2, reference_types_2 = get_all_lists_of_stance_features_in_sent(stances_2)
    stances_3 = sent_annotations_3['stances']
    confidence_levels_3, stance_types_3, polarities_3, reference_names_3, reference_types_3 = get_all_lists_of_stance_features_in_sent(
        stances_3)

    stance_polarity_indications_1 = sent_annotations_1["stance_polarity_indications"]
    stance_polarity_indication_sources_1 = get_all_lists_of_stance_polarity_sources_features_in_sent(stance_polarity_indications_1)
    stance_polarity_indications_2 = sent_annotations_2["stance_polarity_indications"]
    stance_polarity_indication_sources_2 = get_all_lists_of_stance_polarity_sources_features_in_sent(
        stance_polarity_indications_2)
    stance_polarity_indications_3 = sent_annotations_3["stance_polarity_indications"]
    stance_polarity_indication_sources_3 = get_all_lists_of_stance_polarity_sources_features_in_sent(
        stance_polarity_indications_3)

    if keep_stance_type:
        all_stance_features_list_1 = list(itertools.chain(confidence_levels_1, stance_types_1, polarities_1, reference_names_1 , reference_types_1, stance_polarity_indication_sources_1))
        all_stance_features_list_2 = list(itertools.chain(confidence_levels_2, stance_types_2, polarities_2, reference_names_2 , reference_types_2, stance_polarity_indication_sources_1))
        all_stance_features_list_3 = list(itertools.chain(confidence_levels_3, stance_types_3, polarities_3, reference_names_3 , reference_types_3, stance_polarity_indication_sources_1))
    else:
        all_stance_features_list_1 = list(
            itertools.chain(confidence_levels_1, polarities_1, reference_names_1, reference_types_1,
                            stance_polarity_indication_sources_1))
        all_stance_features_list_2 = list(
            itertools.chain(confidence_levels_2, polarities_2, reference_names_2, reference_types_2,
                            stance_polarity_indication_sources_1))
        all_stance_features_list_3 = list(
            itertools.chain(confidence_levels_3, polarities_3, reference_names_3, reference_types_3,
                            stance_polarity_indication_sources_1))

    all_stance_features_sent_agreement, all_stance_features_agreement_record = check_sent_feature_list_agreement(
        all_stance_features_list_1,
        all_stance_features_list_2,
        all_stance_features_list_3,
        sent_annotations_1["sent_text"],
        sent_name=sent_name)

    all_stance_features_agreement_record.pop("Shira_annotations")
    all_stance_features_agreement_record.pop("Avia_annotations")
    all_stance_features_agreement_record.pop("Israel_annotations")

    all_stance_features_agreement_record["Shira_confidence_levels"] = confidence_levels_1
    all_stance_features_agreement_record["Avia_confidence_levels"] = confidence_levels_2
    all_stance_features_agreement_record["Israel_confidence_levels"] = confidence_levels_3

    if keep_stance_type:
        all_stance_features_agreement_record["Shira_stance_types"] = stance_types_1
        all_stance_features_agreement_record["Avia_stance_types"] = stance_types_2
        all_stance_features_agreement_record["Israel_stance_types"] = stance_types_3


    all_stance_features_agreement_record["Shira_polarity"] = polarities_1
    all_stance_features_agreement_record["Avia_polarity"] = polarities_2
    all_stance_features_agreement_record["Israel_polarity"] = polarities_3

    all_stance_features_agreement_record["Shira_reference_names"] = reference_names_1
    all_stance_features_agreement_record["Avia_reference_names"] = reference_names_2
    all_stance_features_agreement_record["Israel_reference_names"] = reference_names_3

    all_stance_features_agreement_record["Shira_reference_types"] = reference_types_1
    all_stance_features_agreement_record["Avia_reference_types"] = reference_types_2
    all_stance_features_agreement_record["Israel_reference_types"] = reference_types_3

    all_stance_features_agreement_record["Shira_polarity_indication_sources"] = stance_polarity_indication_sources_1
    all_stance_features_agreement_record["Avia_polarity_indication_sources"] = stance_polarity_indication_sources_2
    all_stance_features_agreement_record["Israel_polarity_indication_sources"] = stance_polarity_indication_sources_3


    if ALL_FEATURES_AGREEMENT_IN_ONE_FILE:
        return all_stance_features_list_1, all_stance_features_list_2, all_stance_features_list_3, all_stance_features_agreement_record
    else:
        return all_stance_features_sent_agreement, all_stance_features_agreement_record

#here
def get_sent_all_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name):
    all_check_worthiness_features_list_1, all_check_worthiness_features_list_2, all_check_worthiness_features_list_3, all_check_worthiness_features_agreement_record = get_sent_all_check_worthiness_features_agreement(sent_annotations_1, sent_annotations_2,sent_annotations_3,sent_name)

    all_stance_features_list_1, all_stance_features_list_2, all_stance_features_list_3, all_stance_features_agreement_record = get_sent_all_stance_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name)
    all_agency_and_esp_features_list_1, all_agency_and_esp_features_list_2, all_agency_and_esp_features_list_3,all_agency_and_esp_features_agreement_record =  get_sent_all_agency_and_esp_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name)

    all_quantities_features_list_1, all_quantities_features_list_2, all_quantities_features_list_3, all_quantities_features_agreement_record=get_sent_all_quantities_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name)

    all_time_expressions_features_list_1, all_time_expressions_features_list_2, all_time_expressions_features_list_3, all_time_expressions_features_agreement_record= get_sent_all_time_expressions_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name)



    all_features_list_1 = list(itertools.chain(all_check_worthiness_features_list_1, all_stance_features_list_1, all_agency_and_esp_features_list_1, all_quantities_features_list_1, all_time_expressions_features_list_1))
    all_features_list_2 = list(itertools.chain(all_check_worthiness_features_list_2, all_stance_features_list_2, all_agency_and_esp_features_list_2, all_quantities_features_list_2, all_time_expressions_features_list_2))
    all_features_list_3 = list(itertools.chain(all_check_worthiness_features_list_3, all_stance_features_list_3 , all_agency_and_esp_features_list_3, all_quantities_features_list_3, all_time_expressions_features_list_3))

    all_features_sent_agreement, all_features_agreement_record = check_sent_feature_list_agreement(
        all_features_list_1,
        all_features_list_2,
        all_features_list_3,
        sent_annotations_1["sent_text"],
        sent_name=sent_name)

    all_features_agreement_record.pop("Shira_annotations")
    all_features_agreement_record.pop("Avia_annotations")
    all_features_agreement_record.pop("Israel_annotations")

    all_features_agreement_record = all_features_agreement_record | all_check_worthiness_features_agreement_record
    all_features_agreement_record = all_features_agreement_record | all_stance_features_agreement_record
    all_features_agreement_record = all_features_agreement_record | all_agency_and_esp_features_agreement_record
    all_features_agreement_record = all_features_agreement_record | all_quantities_features_agreement_record
    all_features_agreement_record = all_features_agreement_record | all_time_expressions_features_agreement_record
    return all_features_sent_agreement, all_features_agreement_record


def get_sent_all_check_worthiness_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_name):
    sent_check_worthiness_1 = sent_annotations_1['check_worthiness']
    check_worthiness_score_1, claim_type_1, factuality_profile_source_1, factuality_profile_value_1 =get_all_lists_of_checkworthiness_features_in_sent(sent_check_worthiness_1)
    sent_check_worthiness_2 = sent_annotations_2['check_worthiness']
    check_worthiness_score_2, claim_type_2, factuality_profile_source_2, factuality_profile_value_2 =get_all_lists_of_checkworthiness_features_in_sent(sent_check_worthiness_2)
    sent_check_worthiness_3 = sent_annotations_3['check_worthiness']
    check_worthiness_score_3, claim_type_3, factuality_profile_source_3, factuality_profile_value_3 =get_all_lists_of_checkworthiness_features_in_sent(sent_check_worthiness_3)


    all_check_worthiness_features_list_1 = list(itertools.chain(check_worthiness_score_1, claim_type_1, factuality_profile_source_1,factuality_profile_value_1 ))
    all_check_worthiness_features_list_2 = list(itertools.chain(check_worthiness_score_2, claim_type_2, factuality_profile_source_2,factuality_profile_value_2 ))
    all_check_worthiness_features_list_3 = list(itertools.chain(check_worthiness_score_3, claim_type_3, factuality_profile_source_3,factuality_profile_value_3 ))

    all_check_worthiness_features_sent_agreement, all_check_worthiness_features_agreement_record = check_sent_feature_list_agreement(
        all_check_worthiness_features_list_1,
        all_check_worthiness_features_list_2,
        all_check_worthiness_features_list_3,
        sent_annotations_1["sent_text"],
        sent_name=sent_name)

    all_check_worthiness_features_agreement_record.pop("Shira_annotations")
    all_check_worthiness_features_agreement_record.pop("Avia_annotations")
    all_check_worthiness_features_agreement_record.pop("Israel_annotations")

    all_check_worthiness_features_agreement_record["Shira_checkworthiness_score"] = check_worthiness_score_1
    all_check_worthiness_features_agreement_record["Avia_checkworthiness_score"] = check_worthiness_score_2
    all_check_worthiness_features_agreement_record["Israel_checkworthiness_score"] = check_worthiness_score_3


    all_check_worthiness_features_agreement_record["Shira_claim_type"] = claim_type_1
    all_check_worthiness_features_agreement_record["Avia_claim_type"] = claim_type_2
    all_check_worthiness_features_agreement_record["Israel_claim_type"] = claim_type_3


    all_check_worthiness_features_agreement_record["Shira_factuality_profile_source"] = factuality_profile_source_1
    all_check_worthiness_features_agreement_record["Avia_factuality_profile_source"] = factuality_profile_source_2
    all_check_worthiness_features_agreement_record["Israel_factuality_profile_source"] = factuality_profile_source_3


    all_check_worthiness_features_agreement_record["Shira_factuality_profile"] = factuality_profile_value_1
    all_check_worthiness_features_agreement_record["Avia_factuality_profile"] = factuality_profile_value_2
    all_check_worthiness_features_agreement_record["Israel_factuality_profile"] = factuality_profile_value_3





    if ALL_FEATURES_AGREEMENT_IN_ONE_FILE:
        return all_check_worthiness_features_list_1, all_check_worthiness_features_list_2, all_check_worthiness_features_list_3,all_check_worthiness_features_agreement_record
    else:
        return all_check_worthiness_features_sent_agreement, all_check_worthiness_features_agreement_record

def get_all_lists_of_time_exp_features_in_sent(sent_time_exp_layer):
    original_time_tokens_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_time_exp_layer, 'original_time_tokens')
    formatted_timeEXP_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_time_exp_layer, 'formatted_timeEXP')
    return original_time_tokens_list, formatted_timeEXP_list

def get_all_lists_of_time_range_exp_features_in_sent(sent_time_range_exp_layer):
    original_range_time_tokens_list  = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_time_range_exp_layer, 'original_time_tokens')
    formatted_begin_date_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_time_range_exp_layer, 'formatted_begin_date')
    formatted_end_date_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_time_range_exp_layer, 'formatted_end_date')
    return original_range_time_tokens_list, formatted_begin_date_list, formatted_end_date_list
def get_all_lists_of_quantities_features_in_sent(sent_quantities_layer):
    expressions_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_quantities_layer, 'expression')
    quantifiers_types_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_quantities_layer, 'quantifier_type')
    accuracy_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_quantities_layer, 'accuracy')
    the_whole_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_quantities_layer, 'the_whole')
    return expressions_list, quantifiers_types_list, accuracy_list, the_whole_list

def get_all_lists_of_agents_of_predicates_relation_features_in_sent(agents_of_predicates_relation_layer):
    agent_source_predicate_source_relations_list = get_list_of_all_values_of_agents_of_predicates_relation_layer_in_sent(agents_of_predicates_relation_layer)
    return agent_source_predicate_source_relations_list
def get_all_lists_of_agency_predicate_features_in_sent(sent_agency_predicates_layer):
    agency_predicates_sources_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agency_predicates_layer, 'source')
    return agency_predicates_sources_list
def get_all_lists_of_agency_features_in_sent(sent_agencies_layer):
    agency_sources_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'source')
    agency_types_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'agency_type')
    agentless_reasons_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'agentless_reason')
    position_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'position')
    animacy_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'animacy' )
    morphologies_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_agencies_layer, 'morphology')
    return agency_sources_list, agency_types_list, agentless_reasons_list, position_list, animacy_list, morphologies_list

def get_all_lists_of_stance_polarity_sources_features_in_sent(stance_polarity_indications_layer):
    stance_polarity_indications_sources = get_list_of_all_values_of_certain_layer_feature_in_sent(stance_polarity_indications_layer, "source")
    return stance_polarity_indications_sources
def get_all_lists_of_stance_features_in_sent(sent_stances_layer):
    confidence_levels_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_stances_layer,
                                                                                       'confidence_level')
    stance_types_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_stances_layer,
                                                                                       'stance_type')
    polarities_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_stances_layer,
                                                                                       'polarity')
    reference_names_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_stances_layer,
                                                                                       'reference_name')
    reference_types_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_stances_layer,
                                                                                       'reference_type')
    return confidence_levels_list, stance_types_list, polarities_list, reference_names_list, reference_types_list
def get_all_lists_of_checkworthiness_features_in_sent(sent_check_worthiness_layer):
    check_worthiness_score_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_check_worthiness_layer,
                                                                                       "check_worthiness_score")
    if IGNORE_NOT_A_FACTUAL_PROPOSITION:
        for score, i in zip(check_worthiness_score_list, range(len(check_worthiness_score_list))):
            if score == "not a factual proposition":
                check_worthiness_score_list[i] = 'not worth checking'

    claim_type_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_check_worthiness_layer, 'claim_type')
    if IGNORE_NOT_A_CLAIM:
        for claim_type, i in zip(claim_type_list, range(len(claim_type_list))):
            if claim_type == 'not a claim':
                claim_type_list[i] = 'other type of claim'
    factuality_profile_source_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_check_worthiness_layer,
                                                                                          'factuality_profile_source')
    factuality_profile_value_list = get_list_of_all_values_of_certain_layer_feature_in_sent(sent_check_worthiness_layer,
                                                                                         'factuality_profile_value')
    return check_worthiness_score_list, claim_type_list, factuality_profile_source_list, factuality_profile_value_list


def get_list_of_all_values_of_certain_layer_feature_in_sent(sent_layer, feature_name):
    return [d[feature_name] for d in sent_layer if
            feature_name in d]

def get_list_of_all_values_of_agents_of_predicates_relation_layer_in_sent(layer):
    values = []
    for d in layer:
        agent_source = d["agent_source"]
        predicate_source = d["predicate_source"]
        values.append(f'agent_source: {agent_source}, predicate_source: {predicate_source}')
    return values
def get_sent_check_worthiness_agreement(sent_annotations_1, sent_annotations_2, sent_annotations_3, sent_name, type_of_dissagreement_counter):
    sent_check_worthiness_1 = sent_annotations_1['check_worthiness']
    check_worthiness_score_1 = [d['check_worthiness_score'] for d in sent_check_worthiness_1 if
                                'check_worthiness_score' in d]
    sent_check_worthiness_2 = sent_annotations_2['check_worthiness']
    check_worthiness_score_2 = [d['check_worthiness_score'] for d in sent_check_worthiness_2 if
                                'check_worthiness_score' in d]
    sent_check_worthiness_3 = sent_annotations_3['check_worthiness']
    check_worthiness_score_3 = [d['check_worthiness_score'] for d in sent_check_worthiness_3 if
                                'check_worthiness_score' in d]
    if IGNORE_NOT_A_FACTUAL_PROPOSITION:
        for score, i in zip(check_worthiness_score_1, range(len(check_worthiness_score_1))):
            if score == "not a factual proposition":
                check_worthiness_score_1[i] = 'not worth checking'
        for score, i in zip(check_worthiness_score_2, range(len(check_worthiness_score_2))):
            if score == "not a factual proposition":
                check_worthiness_score_2[i] = 'not worth checking'
        for score, i in zip(check_worthiness_score_3, range(len(check_worthiness_score_3))):
            if score == "not a factual proposition":
                check_worthiness_score_3[i] = 'not worth checking'
    sent_agreement, agreement_record = check_sent_feature_list_agreement(check_worthiness_score_1,
                                                                         check_worthiness_score_2,
                                                                         check_worthiness_score_3,
                                                                         sent_annotations_1["sent_text"],
                                                                         sent_name=sent_name,
                                                                         counter=type_of_dissagreement_counter)
    return sent_agreement, agreement_record


# def get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter):
    sorted_agreement_score_counter = {k: v for k, v in
                                      sorted(agreement_score_counter.items(), key=lambda item: item[1], reverse=True)}
    print(f"agreement score sentences counter:")
    for key in sorted_agreement_score_counter:
        print(f'{key}: {sorted_agreement_score_counter[key]}')
    return sorted_agreement_score_counter


def get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences = 100):
    keys = list(agreement_score_counter.keys())
    keys.sort()
    perc_agreement_score_counter = {}
    for key in keys:
        perc_score = (agreement_score_counter[key] / total_num_of_sentences) * 100
        perc_agreement_score_counter[key] = perc_score
    return perc_agreement_score_counter


def get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter):
    sorted_perc_agreement_score_counter = {k: v for k, v in
                                           sorted(perc_agreement_score_counter.items(), key=lambda item: item[1],
                                                  reverse=True)}
    print(f"agreement score sentences percentage:")
    for key in sorted_perc_agreement_score_counter:
        print(f'{key}: {sorted_perc_agreement_score_counter[key]:.2f}%')



def get_and_print_sorted_type_of_dissagreement_counter(type_of_dissagreement_counter):
    sorted_type_of_dissagreement_counter = {k: v for k, v in
                                            sorted(type_of_dissagreement_counter.items(), key=lambda item: item[1],
                                                   reverse=True)}
    print("number of times for each dissagreement:")
    for key in sorted_type_of_dissagreement_counter:
        print(f'{key}: {sorted_type_of_dissagreement_counter[key]}')


def get_all_time_expressions_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    sentences_to_analyze.append('dummy')
    total_num_of_sentences_tested = 0
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")
        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue
        sent_agreement, agreement_record = get_sent_all_time_expressions_features_agreement(sent_annotations_1,
                                                                                      sent_annotations_2,
                                                                                      sent_annotations_3,
                                                                                      sent_annotated_1_name)
        total_num_of_sentences_tested +=1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    print("Time Expressions Features:")
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_all_time_expressions_features.csv", index=False)

def get_all_quantities_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    total_num_of_sentences_tested = 0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")
        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue
        sent_agreement, agreement_record = get_sent_all_quantities_features_agreement(sent_annotations_1,
                                                                                          sent_annotations_2,
                                                                                          sent_annotations_3,
                                                                                          sent_annotated_1_name)
        total_num_of_sentences_tested += 1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    print('Quantities Features:')
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_all_quantities_features.csv", index=False)

def get_all_agency_and_esp_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    total_num_of_sentences_tested = 0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")

        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue
        sent_agreement, agreement_record = get_sent_all_agency_and_esp_features_agreement(sent_annotations_1,
                                                                                  sent_annotations_2,
                                                                                  sent_annotations_3,
                                                                                  sent_annotated_1_name)
        total_num_of_sentences_tested += 1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    print("Agency and ESP Features:")
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_all_agency_and_esp_features.csv", index=False)

def get_all_stance_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    total_num_of_sentences_tested =0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")

        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue
        sent_agreement, agreement_record = get_sent_all_stance_features_agreement(sent_annotations_1,
                                                                                            sent_annotations_2,
                                                                                            sent_annotations_3,
                                                                                            sent_annotated_1_name)
        total_num_of_sentences_tested += 1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    print("Stance Features:")
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_all_stance_features.csv", index=False)


def print_majority_score(agreement_record, feature_name):
    if agreement_record[f'Shira_{feature_name}'] == agreement_record[f'Avia_{feature_name}'] == agreement_record[f'Israel_{feature_name}']:
        print(agreement_record[f'Shira_{feature_name}'])
    elif agreement_record[f'Shira_{feature_name}'] == agreement_record[f'Avia_{feature_name}']:
        print(agreement_record[f'Shira_{feature_name}'])
    elif agreement_record[f'Shira_{feature_name}'] == agreement_record[f'Israel_{feature_name}']:
        print(agreement_record[f'Shira_{feature_name}'])
    elif agreement_record[f'Avia_{feature_name}'] == agreement_record[f'Israel_{feature_name}']:
        print(agreement_record[f'Avia_{feature_name}'])
    else:
        print(f'no majority')


def calc_mean_pairwise_agreement(df, feature_column,new_annotator_values = []):
    shira_feature_name = f'Shira_{feature_column}'
    avia_feature_name = f'Avia_{feature_column}'
    israel_feature_name = f'Israel_{feature_column}'
    shira_avia_agreement_counter = 0
    shira_israel_agreement_counter = 0
    avia_israel_agreement_counter = 0
    if new_annotator_values:
        new_shira_agreement_counter = 0
        new_avia_agreement_counter = 0
        new_israel_agreement_counter = 0
    total_num_of_sentences = 0
    import ast

    shira_values = df[shira_feature_name].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]).tolist()
    avia_values = df[avia_feature_name].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]).tolist()
    israel_values = df[israel_feature_name].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]).tolist()

    for index, row in df.iterrows():
        total_num_of_sentences += 1
        shira_val = row[shira_feature_name]
        avia_val = row[avia_feature_name]
        israel_val = row[israel_feature_name]
        if new_annotator_values:
            new_val = new_annotator_values[index]
            new_val = new_val.replace("-","").strip().replace("'","").replace("`","").strip()

            if shira_val.replace("['","").replace("']","").strip() == new_val:
                new_shira_agreement_counter += 1
            if avia_val.replace("['","").replace("']","").strip() == new_val:
                new_avia_agreement_counter += 1
            if israel_val.replace("['","").replace("']","").strip() == new_val:
                new_israel_agreement_counter += 1
        if shira_val == avia_val:
            shira_avia_agreement_counter += 1
        if shira_val == israel_val:
            shira_israel_agreement_counter += 1
        if avia_val == israel_val:
            avia_israel_agreement_counter += 1


    mean_pairwise_agreement_in_num_of_sentences = (shira_avia_agreement_counter +shira_israel_agreement_counter +avia_israel_agreement_counter)/3
    print(f'mean_pairwise_agreement_in_num_of_sentences: {mean_pairwise_agreement_in_num_of_sentences}')
    mean_pairwise_agreement_percentage_of_sentences = ((shira_avia_agreement_counter +shira_israel_agreement_counter +avia_israel_agreement_counter)/total_num_of_sentences)/3
    print(f'mean_pairwise_agreement_percentage_of_sentences: {mean_pairwise_agreement_percentage_of_sentences}')
    shira_pairwise_agreement_score = (shira_avia_agreement_counter +shira_israel_agreement_counter)/2
    print(f'shira_pairwise_agreement_score: {shira_pairwise_agreement_score}. per: {shira_pairwise_agreement_score/total_num_of_sentences}')
    avia_pairwise_agreement_score = (shira_avia_agreement_counter +avia_israel_agreement_counter)/2
    print(f'avia_pairwise_agreement_score: {avia_pairwise_agreement_score}. per: {avia_pairwise_agreement_score/total_num_of_sentences}')
    israel_pairwise_agreement_score = (shira_israel_agreement_counter +avia_israel_agreement_counter)/2
    print(f'israel_pairwise_agreement_score: {israel_pairwise_agreement_score}. per: {israel_pairwise_agreement_score/total_num_of_sentences}')
    if new_annotator_values:
        new_annotator_pairwise_agreement_score = (new_israel_agreement_counter + new_avia_agreement_counter +new_shira_agreement_counter)/3
        print(f'new_annotator_pairwise_agreement_score: {new_annotator_pairwise_agreement_score}. per: {new_annotator_pairwise_agreement_score/total_num_of_sentences}')
    kappa_shira_avia = cohen_kappa_score(shira_values, avia_values)
    kappa_shira_israel = cohen_kappa_score(shira_values, israel_values)
    kappa_avia_israel = cohen_kappa_score(avia_values, israel_values)


    shira_kappa_score = np.mean([kappa_shira_avia, kappa_shira_israel])
    print(f'Shira mean kappa score: {shira_kappa_score}')
    avia_kappa_score = np.mean([kappa_shira_avia, kappa_avia_israel])
    print(f'Avia mean kappa score: {avia_kappa_score}')
    israel_kappa_score = np.mean([kappa_shira_israel, kappa_avia_israel])
    print(f'Israel mean kappa score: {israel_kappa_score}')
    total_annotators_kappa_score = np.mean([kappa_shira_avia, kappa_shira_israel, kappa_avia_israel])
    print(f'mean pair-wise kappa score: {total_annotators_kappa_score}')


    if new_annotator_values:
        kappa_new_shira = cohen_kappa_score(new_annotator_values, shira_values)
        kappa_new_avia = cohen_kappa_score(new_annotator_values, avia_values)
        kappa_new_israel = cohen_kappa_score(new_annotator_values, israel_values)

    if new_annotator_values:
        new_annotator_score = np.mean([kappa_new_shira, kappa_new_avia, kappa_new_israel])
        print(f'new annotator mean pair-wise Kappa score: {new_annotator_score}')



def get_all_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    total_num_of_sentences_tested =0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")
            continue
        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue

        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue
        # print(f"\"{sent_annotations_1['sent_text']}\",")
        sent_agreement, agreement_record = get_sent_all_features_agreement(sent_annotations_1,
                                                                                            sent_annotations_2,
                                                                                            sent_annotations_3,
                                                                                            sent_annotated_1_name)
        total_num_of_sentences_tested += 1
        # if sent_agreement < 3:
        #     sentences_to_analyze.append(agreement_record)
        sentences_to_analyze.append(agreement_record)
        print_majority_score(agreement_record, 'checkworthiness_score')
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested )
    print(f'total_num_of_sents {len(annotator_1_sentences)}')
    print(f'num_of_sents_tested {total_num_of_sentences_tested}')
    print("Check all features :")
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    calc_mean_pairwise_agreement(df, "checkworthiness_score")
    file_name = "all_features_agreement"
    if ONLY_SAME_NUM_OF_CLAIMS and not ONLY_ONE_CLAIM and not ONLY_SAME_CLAIMS:
        file_name += "_only_same_num_of_claims"
    if ONLY_SAME_CLAIMS and not ONLY_ONE_CLAIM:
        file_name += "_only_same_claims"
    if ONLY_ONE_CLAIM:
        file_name +="_only_one_claim"
    if IGNORE_NOT_A_FACTUAL_PROPOSITION:
        file_name += "_ignore_not_a_factual_proposition"
    if IGNORE_NOT_A_CLAIM:
        file_name += "_ignore_not_a_claim"

    df.to_csv(f"{file_name}.csv", index=False, encoding="utf-8")
    df_transposed = df.transpose()
    df_transposed.to_csv(f"transposed_{file_name}.csv", index=True)
    return file_name
def get_all_check_worthiness_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    sentences_to_analyze = []
    total_num_of_sentences_tested =0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)

        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)

        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")


        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue

        sent_agreement, agreement_record = get_sent_all_check_worthiness_features_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_annotated_1_name)
        total_num_of_sentences_tested += 1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    print("Check Worthiness Features:")
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_all_check_worthiness_features.csv", index=False)

def get_check_worthiness_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences):
    agreement_score_counter = {}  # key:1-no agreement, 2- two agreed, 3 - three agreed :sentence_counter
    type_of_dissagreement_counter = {}
    type_of_dissagreement_counter["dummy"] = 0
    sentences_to_analyze = []
    total_num_of_sentences_tested =0
    sentences_to_analyze.append('dummy')
    for sent_annotated_1_name, sent_annotated_2_name, sent_annotated_3_name in zip(annotator_1_sentences,
                                                                                   annotator_2_sentences,
                                                                                   annotator_3_sentences):
        assert (sent_annotated_1_name == sent_annotated_2_name == sent_annotated_3_name)
        sent_annotations_1, sent_annotations_2, sent_annotations_3 = get_sent_annotators_annotations(
            annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path,
            annotator_3_sentences_annotations_path, sent_annotated_1_name)
        if not (sent_annotations_1["sent_text"] == sent_annotations_2["sent_text"] and sent_annotations_2[
            'sent_text'] ==
                sent_annotations_3['sent_text']):
            print("not same sentence!!")
        if ONLY_SAME_NUM_OF_CLAIMS:
            if len(sent_annotations_1['check_worthiness']) != len(sent_annotations_2['check_worthiness']) or len(sent_annotations_1['check_worthiness'])!= len(sent_annotations_3['check_worthiness']):
                continue
            if ONLY_SAME_CLAIMS:
                not_same_claims = False
                for i in range(len(sent_annotations_1['check_worthiness'])):
                    start_claim_tokens_1 = sent_annotations_1['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_2 = sent_annotations_2['check_worthiness'][i]['start_claim_token']
                    start_claim_tokens_3 = sent_annotations_3['check_worthiness'][i]['start_claim_token']
                    if start_claim_tokens_1 != start_claim_tokens_2 or start_claim_tokens_1 != start_claim_tokens_3:
                        not_same_claims = True
                if not_same_claims:
                    continue
        if ONLY_ONE_CLAIM:
            if len(sent_annotations_1['check_worthiness']) != 1 or len(sent_annotations_2['check_worthiness']) != 1 or len(sent_annotations_3['check_worthiness']) != 1:
                continue

        sent_agreement, agreement_record = get_sent_check_worthiness_agreement(sent_annotations_1, sent_annotations_2,
                                                                               sent_annotations_3,
                                                                               sent_annotated_1_name, type_of_dissagreement_counter)
        total_num_of_sentences_tested += 1
        if sent_agreement < 3:
            sentences_to_analyze.append(agreement_record)
        count = agreement_score_counter.get(sent_agreement, 0)
        count += 1
        agreement_score_counter[sent_agreement] = count
    type_of_dissagreement_counter.pop("dummy")
    sentences_to_analyze.remove("dummy")
    perc_agreement_score_counter = get_percentage_agreement_score_counter(agreement_score_counter, total_num_of_sentences_tested)
    # get_and_print_sorted_agreement_score_sentences_counter(agreement_score_counter)
    get_and_print_sorted_perc_agreement_counter(perc_agreement_score_counter)
    get_and_print_sorted_type_of_dissagreement_counter(type_of_dissagreement_counter)
    df = pd.DataFrame.from_dict(sentences_to_analyze)
    df.to_csv("low_agreement_sentences_on_check_worthiness.csv", index=False)


if __name__ == '__main__':
    # keep_only_same_files_in_each_annotator_dir("C:\\data\\gili\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\second_try_annotations_agreement\\avia", "C:\\data\\gili\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\second_try_annotations_agreement\\israel", "C:\\data\\gili\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\second_try_annotations_agreement\\shira")
    annotator_name_1 = Annotator_Name.SHIRA.value
    annotator_name_2 = Annotator_Name.AVIA.value
    annotator_name_3 = Annotator_Name.ISRAEL.value
    annotator_1_sentences_annotations_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH,'annotatitions_for_agreement_statistics','parsed', annotator_name_1)
    annotator_2_sentences_annotations_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH, 'annotatitions_for_agreement_statistics','parsed',annotator_name_2)
    annotator_3_sentences_annotations_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH, 'annotatitions_for_agreement_statistics','parsed', annotator_name_3)
    annotator_1_sentences, annotator_2_sentences, annotator_3_sentences = get_annotators_sorted_sentences_names(annotator_1_sentences_annotations_path, annotator_2_sentences_annotations_path, annotator_3_sentences_annotations_path)

    get_check_worthiness_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)

    saved_file_name = get_all_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)
    df = pd.read_csv(f"{saved_file_name}.csv")

    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4_check_worthiness_score.txt"
    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4o_check_worthiness_score.txt"

    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4_check_worthiness_score hebrew prompt english labels.txt"
    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4o_check_worthiness_score hebrew prompt english labels.txt"

    gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4_check_worthiness_score hebrew prompt and labels.txt"
    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4o_check_worthiness_score hebrew prompt and labels.txt"

    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4_check_worthiness_score_with_explanations.txt"
    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4o_check_worthiness_score_with_explanations.txt"

    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4_check_worthiness_score_with_real_examples.txt"
    # gpt_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\דוקטורט\\factuality\gpt_annotations\only_one_claim_temp_0_gpt4o_check_worthiness_score_with_real_examples.txt"
    knesset_dicta_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\\דוקטורט\\factuality\\finetuned_models_annotations\\knesset-dicta-results-on-one-claim-test-set.txt"
    dictabert_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\\דוקטורט\\factuality\\finetuned_models_annotations\\dictabert-results-on-one-claim-test-set.txt"
    alephbertgimmel_answers_file = "G:\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\OurDrive\\University of Haifa\\דוקטורט\\factuality\\finetuned_models_annotations\\alephbertgimmel-results-on-one-claim-test-set.txt"
    # answers_files = gpt_answers_file
    answers_files = dictabert_answers_file
    with open(answers_files, encoding="utf-8") as gpt_file:
        gpt_annotations = gpt_file.readlines()
        gpt_annotations.pop(0)
        gpt_annotations = [x.strip().strip('"').strip("'").strip("-").lower() for x in gpt_annotations if x.strip()!="" ]
        gpt_annotations = [x.replace("לא טענה עובדתית", "not a factual proposition").replace("לא שווה לבדוק", "not worth checking").replace("שווה לבדוק","worth checking") for x in gpt_annotations if x.strip()!="" ]
    calc_mean_pairwise_agreement(df, "checkworthiness_score", gpt_annotations)

    # get_all_check_worthiness_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)
    # get_all_stance_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)
    # get_all_agency_and_esp_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)
    # get_all_quantities_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)
    # get_all_time_expressions_features_agreement_on_all_sentences(annotator_1_sentences, annotator_2_sentences, annotator_3_sentences)






