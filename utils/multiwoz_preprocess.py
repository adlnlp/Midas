"""
This code is from https://github.com/jasonwu0731/trade-dst
"""


def fix_general_label_error(labels, type, slots):
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels]) 

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        
        # added by us
        "do n't care":"dontcare", "the castle galleries":"castle galleries", "the golden house":"golden house",
        "efes":"efes restaurant", "boating":"boat"
        
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
            
            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict

"""
This code is from https://github.com/smartyfh/DST-MetaASSIST/blob/main/STAR/preprocess_data.py
"""

import json
import os
import re
import argparse

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

data_files = ["train_dials.json", "dev_dials.json", "test_dials.json"]

#--------------------------------
def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text

def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("anthropogy", "anthropology", text)
    text = re.sub("theater", "theatre", text)
    text = re.sub("musuem", "museum", text)
    text = re.sub("the weat", "the west", text)
    text = re.sub("the wast", "the west", text)
    text = re.sub(" wendesday ", " wednesday ", text)
    text = re.sub(" wednes ", " wednesday ", text)
    text = re.sub("thurtsday", "thursday", text)
    text = re.sub("mdoerate", "moderate", text)
    text = re.sub("portugese", "portuguese", text)
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    return text

def normalize_label(slot, value_label):
    value_label = re.sub(">", "|", value_label)
    if "|" in value_label:
        temp_value = sorted(value_label.split("|"))
        value_label = "|".join(temp_value)
        
    # Some general typos
    value_label = re.sub("theater", "theatre", value_label)
    value_label = re.sub("archaelogy", "archaeology", value_label)
    value_label = re.sub("anthropogy", "anthropology", value_label)
    value_label = re.sub("portugese", "portuguese", value_label)
    value_label = re.sub("concerthall", "concert hall", value_label)
    
    # Normalization of empty slots
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveat" in slot or "arriveby" in slot or slot == 'restaurant-book time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)
        
    if slot == "hotel-name":
        value_label = re.sub("b & b", "bed and breakfast", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes' or value_label == 'free':
            return "true"
        if value_label == "no":
            return "false"
    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label
#--------------------------------

def make_slot_meta(ontology):
    meta = []
    change = {}
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        meta.append('-'.join([d, s.lower()]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


def preprocess(data_dir = 'data/mwz2.4'):
    ### Read ontology file
    fp_ont = open(os.path.join(data_dir, "ontology.json"), "r")
    data_ont = json.load(fp_ont)
    fp_ont.close()

    slot_meta, _ = make_slot_meta(data_ont)
    ontology_modified = {}
    for slot in slot_meta:
        ontology_modified[slot] = []

    ### normalize text and fix label errors
    for idx, file_id in enumerate(data_files):   
        fp_data = open(os.path.join(data_dir, file_id), "r")
        dials = json.load(fp_data)

        dials_v2 = []
        for dial_dict in dials:
            new_dial_dict = {} 
            new_dial_dict["dialogue_idx"] = dial_dict["dialogue_idx"]
            new_dial_dict["domains"] = dial_dict["domains"]
            new_dial_dict["dialogue"] = []

            prev_turn_state = {}
            for slot in slot_meta:
                prev_turn_state[slot] = "none"
                ontology_modified[slot].append("none")

            tidx = 0
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                if idx == 0 and turn_domain not in EXPERIMENT_DOMAINS: # only training set contains hospital and police domains
                    continue

                turn["turn_idx"] = tidx
                tidx += 1

                turn["system_transcript"] = normalize_text(turn["system_transcript"])
                turn["transcript"] = normalize_text(turn["transcript"])

                # state
                turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
                for slot in slot_meta:
                    if slot not in turn_dialog_state:
                        turn_dialog_state[slot] = "none"
                    else:
                        turn_dialog_state[slot] = normalize_label(slot, turn_dialog_state[slot])

                    if turn_dialog_state[slot]=="dontcare":
                        turn_dialog_state[slot] = "do not care"

                    ontology_modified[slot].append(turn_dialog_state[slot])

                turn["belief_state"] = [] # we go through turn_label to generate the full state

                # turn label
                turn_label = []
                for slot in slot_meta:
                    if turn_dialog_state[slot] != prev_turn_state[slot]:
                        turn_label.append([slot, turn_dialog_state[slot]])
                turn["turn_label"] = turn_label  

                prev_turn_state = turn_dialog_state
                new_dial_dict["dialogue"].append(turn)

            dials_v2.append(new_dial_dict)

        with open(os.path.join(data_dir, file_id.split(".")[0]+"_v2.json"), 'w') as outfile:
            json.dump(dials_v2, outfile, indent=4)

    # ontology extracted from dataset
    for slot in slot_meta:
        ontology_modified[slot] = sorted(list(set(ontology_modified[slot])))

    with open(os.path.join(data_dir, 'ontology-modified.json'), 'w') as outfile:
         json.dump(ontology_modified, outfile, indent=4)
