import json
import re


'''
The codes are modified from https://github.com/budzianowski/multiwoz
'''


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def get_bstate(bstate):
    summary_bstate = []
    summary_bvalue = []

    for slot in bstate:
        slot_enc = [0, 0]    # not mentioned, filled
        if len(bstate[slot]) == 0:
            slot_enc[0] = 1
            summary_bvalue.append([slot.strip().lower(), ""])
            summary_bstate += slot_enc
            continue
        else:
            slot_enc[1] = 1
        summary_bstate += slot_enc

        if isinstance(bstate[slot], list):
            summary_bvalue.append([slot.strip().lower(), normalize(bstate[slot][0].strip().lower())])
        else:
            summary_bvalue.append([slot.strip().lower(), normalize(bstate[slot].strip().lower())])

    return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue):
    """Organize the dialogue into patient and doctor turn"""
    d = dialogue
    if len(d['log']) % 2 != 0:
        print('Odd # of turns, ERROR')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    pat_turns = []
    doc_turns = []
    for i in range(len(d['log'])):
        belief_summary, belief_value_summary = get_bstate(d['log'][i]['slots'])  # Extract belief states and values
        d['log'][i]['belief_summary'] = str(belief_summary)
        d['log'][i]['belief_value_summary'] = belief_value_summary

        if i % 2 == 0:
            pat_turns.append(d['log'][i])
        else:
            doc_turns.append(d['log'][i])
    d_pp['pat_log'] = pat_turns
    d_pp['doc_log'] = doc_turns

    return d_pp


def get_dial(dialogue):
    """Pack the dialogue"""
    dial = []
    d_orig = analyze_dialogue(dialogue)
    if d_orig is None:
        return None
    pat = [t['text'] for t in d_orig['pat_log']]
    doc = [t['text'] for t in d_orig['doc_log']]
    pat_bvs = [t['belief_value_summary'] for t in d_orig['pat_log']]
    doc_bvs = [t['belief_value_summary'] for t in d_orig['doc_log']]

    for item in zip(pat, doc, pat_bvs, doc_bvs):
        dial.append({'pat': item[0], 'doc': item[1], 'pat_bvs': item[2], 'doc_bvs': item[3]})

    return dial


def createData():
    delex_data = {}
    train_dials = []
    dev_dials = []
    test_dials = []

    fin = open("../covid-dst-en-single-value.json", "r", encoding='utf-8')
    data = json.load(fin)
    fin.close()

    fin = open("data/data_split.json", 'r')
    data_split = json.load(fin)
    fin.close()

    count_train, count_dev, count_test = 0, 0, 0

    for didx, dialogue_name in enumerate(data):
        dialogue = data[dialogue_name]
        for idx, turn in enumerate(dialogue["log"]):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn["text"])
            dialogue["log"][idx]["text"] = origin_text
        dial = get_dial(dialogue)
        if dial:
            uttr = {}
            uttr["dialogue_idx"] = dialogue_name
            last_pat_bs = []
            last_doc_bs = []
            uttr["dialogue"] = []

            for turn_idx, turn in enumerate(dial):
                turn_dialog = {}
                turn_dialog["turn_idx"] = turn_idx
                turn_dialog["patient_transcript"] = dial[turn_idx]["pat"]
                turn_dialog["doctor_transcript"] = dial[turn_idx]["doc"]
                turn_dialog["pat_belief_state"] = turn["pat_bvs"]
                turn_dialog["doc_belief_state"] = turn["doc_bvs"]
                turn_dialog["pat_turn_label"] = [bs for bs in turn_dialog["pat_belief_state"] if bs not in last_pat_bs]
                turn_dialog["doc_turn_label"] = [bs for bs in turn_dialog["doc_belief_state"] if bs not in last_doc_bs]
                last_pat_bs = turn_dialog["pat_belief_state"]
                last_doc_bs = turn_dialog["doc_belief_state"]
                uttr["dialogue"].append(turn_dialog)

            if didx in data_split["train"]:
                train_dials.append(uttr)
                count_train += 1
            elif didx in data_split["dev"]:
                dev_dials.append(uttr)
                count_dev += 1
            else:
                test_dials.append(uttr)
                count_test += 1

    print("# of dialogues: Train {}, Dev {}, Test {}".format(count_train, count_dev, count_test))

    # save all dialogues
    with open('data/dev_dials.json', 'w') as f:
        json.dump(dev_dials, f, indent=4)

    with open('data/test_dials.json', 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open('data/train_dials.json', 'w') as f:
        json.dump(train_dials, f, indent=4)


def main():
    print("Process COVID-19 DST dataset. This might take a while.")
    delex_data = createData()


if __name__ == '__main__':
    main()
