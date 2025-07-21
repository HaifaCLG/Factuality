import json
import os.path
import zipfile
from datetime import datetime
from enum import Enum
from cassis import *
from factuality_schema import Factuality_Schema
from scheme_enums_and_classes import *
from config import *



# inception_project_path_1 = "data\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\projects\\factuality-no-consultations-28342097045501723770"
# inception_project_path_2 = "data\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\projects\\factuality-no-consultations-part-211120689765203106275"
inception_project_path = "data\\processed_knesset\\factuality_manual_annotations\\annotatitions_for_agreement_statistics\\projects\\x100_sentences_for_agreement_no_consultations"
# agreement_project_paths = [inception_project_path_1, inception_project_path_2]
agreement_project_paths = [inception_project_path]
# projects_dir_path = "data\\processed_knesset\\factuality_manual_annotations\\rest_annotations\\projects"
projects_dir_path = "data\\processed_knesset\\factuality_manual_annotations\\final_projects"

def update_protocol_anchors(factualiy_obj, xmi_anchors, text):
    for xmi_anchor in xmi_anchors:
        anchor = This_Protocol_Anchor()
        begin = xmi_anchor.begin
        end = xmi_anchor.end
        source_str = text[int(begin):int(end)].strip()
        if source_str:
            anchor.this_protocol_anchor = True
            anchor.source = (source_str, begin, end)
        factualiy_obj.this_protocol_anchors.append(anchor)



def update_agency(factualiy_obj, xmi_agencies, agents_of_predicates, text):
    for xmi_agency in xmi_agencies:
        if xmi_agency.agency_predicate:
            agency_predicate = Agency_Predicate()
            begin = xmi_agency.begin
            end = xmi_agency.end
            source_str = text[int(begin):int(end)].strip()
            agency_predicate.source = (source_str, begin, end)
            factualiy_obj.agency_predicates.append(agency_predicate)
        else:
            agency = Agency()
            agency.agency_type = xmi_agency.agentexperiencer
            if len(xmi_agency.position.elements) > 1:
                print("hey more than one! ")
            if xmi_agency.position.elements:
                agency.position = xmi_agency.position.elements[0]
            if xmi_agency.animacy.elements:
                agency.animacy = xmi_agency.animacy.elements[0]
            if xmi_agency.morphology.elements:
                agency.morphology = xmi_agency.morphology.elements[0]
            begin = xmi_agency.begin
            end = xmi_agency.end
            source_str = text[int(begin):int(end)].strip()
            agency.source = (source_str, begin, end)
            factualiy_obj.agencies.append(agency)
    for xmi_relation in agents_of_predicates:
        agent_of_predicate = Agent_of_Predicate()
        dependent_begin = xmi_relation.Dependent.begin
        dependent_end = xmi_relation.Dependent.end
        dependent_str = text[int(dependent_begin):int(dependent_end)].strip()
        governor_begin = xmi_relation.Governor.begin
        governor_end = xmi_relation.Governor.end
        governor_str = text[int(governor_begin):int(governor_end)].strip()





        dependant_is_predicate = False
        for predicate in factualiy_obj.agency_predicates:
            if agent_of_predicate.predicate_source == predicate.source:
                dependant_is_predicate = True
                break
        if dependant_is_predicate:
            agent_of_predicate.predicate_source = (dependent_str, dependent_begin, dependent_end)
            agent_of_predicate.agent_source = (governor_str, governor_begin, governor_end)
        else:
            agent_of_predicate.predicate_source = (governor_str, governor_begin, governor_end)
            agent_of_predicate.agent_source = (dependent_str, dependent_begin, dependent_end)

        factualiy_obj.agents_of_predicates.append(agent_of_predicate)


def update_check_worthiness(factuality, xmi_check_worthinesses, xmi_factuality_profiles, xmi_factuality_profile_sources, text):
    if (len(xmi_factuality_profiles) != len(xmi_check_worthinesses)):
        raise Exception("number of factuality profiles not equal to number of checkworthinesses")

    if (len(xmi_factuality_profiles) != len(xmi_factuality_profile_sources)):
        raise Exception("number of factuality profiles not equal to number of xmi_factuality_profile_sources")
    for xmi_check_worthiness, xmi_factuality_profile,xmi_factuality_profile_source, idx in zip(xmi_check_worthinesses, xmi_factuality_profiles,xmi_factuality_profile_sources, range(len(xmi_check_worthinesses))):
        check_worthiness = Check_Worthiness()
        begin = xmi_check_worthiness.begin
        end = xmi_check_worthiness.end
        check_worthiness.start_claim_token = (text[int(begin):int(end)].strip(), begin, end)
        if xmi_check_worthiness.checkworthiness.elements:
            check_worthiness.check_worthiness_score = xmi_check_worthiness.checkworthiness.elements[0]
        else:
            raise Exception("no checkworthinesses score")
        if xmi_check_worthiness.claim_type.elements:
            check_worthiness.claim_type = xmi_check_worthiness.claim_type.elements[0]
        check_worthiness.factuality_profile_source = xmi_factuality_profile_source.source

        if xmi_factuality_profile.factuality_profile_score.elements:
            check_worthiness.factuality_profile_value = xmi_factuality_profile.factuality_profile_score.elements[0]
        factuality.check_worthiness.append(check_worthiness)


def update_stances(factuality, xmi_stances, xmi_stance_refs, text):
    for xmi_stance, idx in zip(xmi_stances, range(len(xmi_stances))):
        stance = Stance()
        stance_polarity_indication = Stance_polarity_indication()
        if xmi_stance.confidencelevel.elements:
            stance.confidence_level = xmi_stance.confidencelevel.elements[0]
        # else:
        #     stance.confidence_level = Stance_Confidence_Level.IRRELEVANT.value
        if xmi_stance.stance_type.elements:
            stance.stance_type = xmi_stance.stance_type.elements[0]
        else:
            stance.stance_type = None
        if xmi_stance.polarity.elements:
            stance.polarity = xmi_stance.polarity.elements[0]
        # else:
        #     stance.polarity = Stance_Polarity.UNDERSPECIFIED.value
        if xmi_stance.polarity_indication:
            if xmi_stance.polarity_indication == "polarity_indication":
                begin = xmi_stance.begin
                end = xmi_stance.end
                stance_polarity_indication.source = (text[int(begin):int(end)].strip(), begin, end)
                factuality.stance_polarity_indications.append(stance_polarity_indication)

        if xmi_stance_refs and len(xmi_stance_refs)>idx:
            if xmi_stance_refs[idx].stance_reference_type.elements:
                stance.reference_type = xmi_stance_refs[idx].stance_reference_type.elements[0]
            begin = xmi_stance_refs[idx].begin
            end = xmi_stance_refs[idx].end
            stance.reference_name = (text[int(begin):int(end)].strip(), begin, end)
            if stance.reference_name == "":
                print("here")
        factuality.stances.append(stance)


def update_hedges(factuality, xmi_hedges, text):
    for xmi_hedge in xmi_hedges:
        begin = xmi_hedge.begin
        end = xmi_hedge.end

        factuality.hedges.append((text[int(begin):int(end)].strip(), begin,end))


def update_quantities(factuality, xmi_quantities, xmi_percentage_ofs, text):

    for xmi_quantity in xmi_quantities:
        quantity = Quantity()
        begin = xmi_quantity.begin
        end = xmi_quantity.end

        quantity.expression = (text[int(begin):int(end)].strip(), begin, end)
        quantity.the_whole = extract_the_whole(begin, end, text, xmi_percentage_ofs)
        if xmi_quantity.quantifierType and xmi_quantity.quantifierType.elements:
            quantity.quantifier_type = xmi_quantity.quantifierType.elements[0]
        if xmi_quantity.exp == "quantities_exp":
            quantity.quantifier_type= Quantifier_Type.QUANTITY_EXP.value

        if xmi_quantity.accuracy.elements:
            quantity.accuracy = xmi_quantity.accuracy.elements[0]

        factuality.quantities.append(quantity)


def extract_the_whole(begin, end, text, xmi_percentage_ofs):
    the_whole = None
    for xmi_perc_of in xmi_percentage_ofs:
        if xmi_perc_of.Governor.begin == begin and xmi_perc_of.Governor.end == end:
            the_whole = (text[int(xmi_perc_of.Dependent.begin):int(xmi_perc_of.Dependent.end)].strip(), xmi_perc_of.Dependent.begin, xmi_perc_of.Dependent.end)
            break
    return the_whole


def update_NERs(factuality, xmi_NERs, text):
    for xmi_ner in xmi_NERs:
        ner = NER()
        if xmi_ner.named_entity_type.elements:
            ner.entity_type = xmi_ner.named_entity_type.elements[0]
        else:
            continue
        begin = xmi_ner.begin
        end = xmi_ner.end
        ner.entity_name = (text[int(begin):int(end)].strip(),begin,end )
        factuality.named_entites.append(ner)


def update_ESPs(factuality, xmi_ESPs, text):
    for xmi_esp in xmi_ESPs:
        esp = ESP_Profile()
        if xmi_esp.ESP_type.elements:
            esp.esp_type = xmi_esp.ESP_type.elements[0]
        else:
            continue
        begin = xmi_esp.begin
        end = xmi_esp.end
        esp.esp = (text[int(begin):int(end)].strip(), begin, end)
        #ADD xmi_esp.ESP which i now change ESP_predicate or ESP_source

        factuality.esp_profiles.append(esp)


def update_time_exps(factuality, xmi_time_exps, text):
    for xmi_time_exp in xmi_time_exps:
        time_exp = Time_EXP_Range()
        begin = xmi_time_exp.begin
        end = xmi_time_exp.end
        time_exp.original_time_tokens = (text[int(begin):int(end)].strip(),begin, end)
        if xmi_time_exp.time_expression_day:
            day = xmi_time_exp.time_expression_day
        else:
            day = None
        if xmi_time_exp.time_expression_month:
            month = xmi_time_exp.time_expression_month
        else:
            month = None
        if xmi_time_exp.time_expression:#year feature is called this way in inception
            year = xmi_time_exp.time_expression
        else:
            year = None
        if xmi_time_exp.time_expression_hour:
            hour = xmi_time_exp.time_expression_hour
        else:
            hour = None
        if xmi_time_exp.time_expression_minute:
            minute = xmi_time_exp.time_expression_minute
        else:
            minute = None
        DATE_FORMAT = '%Y-%m-%d %H:%M'
        date_str = f'{year}-{month}-{day} {hour}:{minute}'
        # datetime_object = datetime.strptime(date_str, DATE_FORMAT)
        time_exp.formatted_begin_date = date_str
        factuality.time_expression_ranges.append(time_exp)



def update_time_exps_ranges(factuality, xmi_time_ranges_exps, text):
    #todo ask about time_expressions_token a_Time_token
    year_start = None
    year_end = None
    month_start = None
    month_end = None
    day_start = None
    day_end = None
    hour_start = None
    hour_end = None
    minute_start = None
    minute_end = None
    for xmi_time_range_exp in xmi_time_ranges_exps:
        time_range_exp = Time_EXP_Range()
        begin = xmi_time_range_exp.begin
        end = xmi_time_range_exp.end
        time_range_exp.original_time_tokens = (text[int(begin):int(end)].strip(), begin, end)
        if xmi_time_range_exp.byear_start.elements:
            year_start  = xmi_time_range_exp.byear_start.elements[0]
        if xmi_time_range_exp.c_year_end.elements:
            year_end = xmi_time_range_exp.c_year_end.elements[0]
        if xmi_time_range_exp.d_month_start.elements:
            month_start = xmi_time_range_exp.d_month_start.elements[0]
        if xmi_time_range_exp.e_month_end.elements:
            month_end = xmi_time_range_exp.e_month_end.elements[0]
        if xmi_time_range_exp.f_day_start.elements:
            day_start = xmi_time_range_exp.f_day_start.elements[0]
        if xmi_time_range_exp.g_day_end.elements:
            day_end = xmi_time_range_exp.g_day_end.elements[0]
        if xmi_time_range_exp.h_hour_start.elements:
            hour_start = xmi_time_range_exp.h_hour_start.elements[0]
        if xmi_time_range_exp.i_hour_end.elements:
            hour_end = xmi_time_range_exp.i_hour_end.elements[0]
        if xmi_time_range_exp.j_minute_start.elements:
            minute_start = xmi_time_range_exp.j_minute_start.elements[0]
        if xmi_time_range_exp.k_minute_end.elements:
            minute_end = xmi_time_range_exp.k_minute_end.elements[0]
        date_start_str = f'{year_start}-{month_start}-{day_start} {hour_start}:{minute_start}'
        date_end_str = f'{year_end}-{month_end}-{day_end} {hour_end}:{minute_end}'
        time_range_exp.formatted_begin_date = date_start_str
        time_range_exp.formatted_end_date = date_end_str
        factuality.time_expression_ranges.append(time_range_exp)


def update_agentless(factuality, agentless_reason_xmi, text):
    for agentless_xmi in agentless_reason_xmi:
        agency = Agency()
        if agentless_xmi:
            agentless_reason = agentless_xmi.agentless.elements

        else:
            agentless_reason = None
        if agentless_reason:
            agency.agency_type = Agency_Type.AGENTLESS.value
            agency.agentless_reason = agentless_reason[0]
        factuality.agencies.append(agency)


def parse_sent_xmi_file(annotator_xmi_file_path, type_system_path):
    # TODO remove from here
    # factuality = Factuality_Schema()
    # with open(type_system_path, 'rb') as f:
    #     typesystem = load_typesystem(f)
    # with open(annotator_xmi_file_path, 'rb') as file:
    #     cas = load_cas_from_xmi(file, typesystem=typesystem)
    # text = cas.get_sofa().sofaString
    # agencies = cas.select('webanno.custom.Agency')
    # agents_of_predicates = cas.select('webanno.custom.Agent_of_predicate')
    # update_agency(factuality, agencies, agents_of_predicates, text)
    # check_worthinesses = cas.select('webanno.custom.Checkworthiness')
    # factuality_profiles = cas.select('webanno.custom.Factuality_profile')
    # factuality_profile_sources = cas.select('webanno.custom.Factuality_profile_source')
    # update_check_worthiness(factuality, check_worthinesses, factuality_profiles, factuality_profile_sources, text)
     # todo remove until here

    with open(type_system_path, 'rb') as f:
        typesystem = load_typesystem(f)
    with open(annotator_xmi_file_path, 'rb') as file:
        cas = load_cas_from_xmi(file, typesystem=typesystem)
    factuality = Factuality_Schema()
    text = cas.get_sofa().sofaString
    factuality.sent_text = text
    validity = cas.select('webanno.custom.Non_valid')
    if validity == "not_a_valid_sentence":
        factuality.non_valid = True
        return factuality
    check_worthinesses = cas.select('webanno.custom.Checkworthiness')
    factuality_profiles = cas.select('webanno.custom.Factuality_profile')
    factuality_profile_sources = cas.select('webanno.custom.Factuality_profile_source')
    update_check_worthiness(factuality, check_worthinesses, factuality_profiles, factuality_profile_sources, text)
    stances = cas.select('webanno.custom.Stance')
    stance_refs = cas.select('webanno.custom.Stance_reference')
    update_stances(factuality, stances,stance_refs, text)
    if 'webanno.custom.This_protocol_anchor' in cas.typesystem._types:
        protocol_anchors = cas.select('webanno.custom.This_protocol_anchor')
        update_protocol_anchors(factuality, protocol_anchors, text)
    hedges = cas.select('webanno.custom.Hedging')
    update_hedges(factuality, hedges, text)
    quantities = cas.select('webanno.custom.Quantities')
    percentage_of = cas.select('webanno.custom.Percentageof')
    update_quantities(factuality, quantities, percentage_of, text)
    agencies = cas.select('webanno.custom.Agency')
    agents_of_predicates = cas.select('webanno.custom.Agent_of_predicate')
    update_agency(factuality, agencies, agents_of_predicates, text)
    if 'webanno.custom.Agentless' in cas.typesystem._types:
        agentless_reason_xmi = cas.select('webanno.custom.Agentless')
    update_agentless(factuality, agentless_reason_xmi, text)
    NERs = cas.select('webanno.custom.Namesentities')
    update_NERs(factuality, NERs, text)
    ESPs = cas.select('webanno.custom.Eventselectingpredicates')
    update_ESPs(factuality, ESPs, text)
    time_ranges_exps = cas.select('webanno.custom.Time_expression_range')
    update_time_exps_ranges(factuality, time_ranges_exps, text)

    return factuality


def save_factuality_as_json_file(sent_factuality, sent_output):
    dir_path = os.path.dirname(os.path.realpath(sent_output))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=False)
    with open(sent_output, 'w', encoding='utf-8') as file:
        json.dump(sent_factuality.__dict__,file, default=lambda o: o.__dict__, ensure_ascii=False)


def parse_all_sentences_in_path(sentences_annotations_path, output_path, annotator_name, print_missing_files = True):
    for sent in os.listdir(sentences_annotations_path):
        sent_output = os.path.join(output_path, f'{sent}.json')
        # if os.path.exists(sent_output):#TODO restore
        #     continue#TODO restore
        sent_path = os.path.join(sentences_annotations_path, sent)
        if os.path.isdir(sent_path):
            annotator_zip_file_name = f'{annotator_name}.zip'
            sent_annotator_zip_path = os.path.join(sent_path, annotator_zip_file_name)
            extracted_path = sent_annotator_zip_path.replace(".zip", "")
            if not os.path.exists(extracted_path):
                if os.path.exists(sent_annotator_zip_path):
                    with zipfile.ZipFile(sent_annotator_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extracted_path)
                else:
                    if print_missing_files:
                        print(f'file does not exist: {extracted_path} or {sent_annotator_zip_path}')
                    continue
            annotator_xmi_file_path = os.path.join(extracted_path, f'{annotator_name}.xmi')
            type_system_path = os.path.join(extracted_path, "TypeSystem.xml")
        else:
            annotator_xmi_file_path = sent_path
            type_system_path = os.path.join(sentences_annotations_path, "TypeSystem.xml")
        try:
            sent_factuality = parse_sent_xmi_file(annotator_xmi_file_path, type_system_path)
        except Exception as e:
            print(f'in sentence {sent} in annotator {annotator_name} there was an exception: {e}')
            continue
        save_factuality_as_json_file(sent_factuality, sent_output)


def parse_all_sentences_for_agreement():
    for annotator in Annotator_Name:
        for inception_project_path in agreement_project_paths:
            annotator_name = annotator.value
            sentences_annotations_path = os.path.join(inception_project_path, "annotation")
            output_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH, "annotatitions_for_agreement_statistics",
                                       "parsed",
                                       annotator_name)
            parse_all_sentences_in_path(sentences_annotations_path, output_path, annotator_name)


def get_annotator_name_from_project_name(project):
    if "avia" in project:
        return Annotator_Name.AVIA.value
    elif "israel" in project:
        return Annotator_Name.ISRAEL.value
    elif "shira" in project:
        return Annotator_Name.SHIRA.value
    else:
        raise ("no annotator found in project name")


def parse_all_individual_projects(projects_dir_path):
    projects_dir = os.listdir(projects_dir_path)
    for project in projects_dir:
        project_path = os.path.join(projects_dir_path, project)
        try:
            annotator_name = get_annotator_name_from_project_name(project)
        except Exception as e:
            continue
        # output_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH, "rest_annotations",
        #                            "parsed",
        #                            annotator_name)
        output_path = os.path.join(FACTUALITY_ANNOTATIONS_PATH, "final_projects",
                                   "parsed_json_annotations",
                                   annotator_name)
        sentences_annotations_path = os.path.join(project_path, "annotation")
        parse_all_sentences_in_path(sentences_annotations_path, output_path, annotator_name, print_missing_files=False)


if __name__ == '__main__':
    # parse_all_sentences_for_agreement()
    parse_all_individual_projects(projects_dir_path)
