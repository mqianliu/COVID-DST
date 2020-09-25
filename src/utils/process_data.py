import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle


ONTOLOGY_PAT = ["cough", "phlegm", "breath", "chest", "runnynose", "throat", "fever", "chills", "pain", "fatigue",
                "headache", "mood", "gastric", "othersym", "medication", "pneumonia", "asthma", "diabetes",
                "otherdiagnosis", "travel", "exposure", "age", "smoking", "otherphycon"]
ONTOLOGY_DOC = ["request", "action", "prescription", "diagnose", "ct/xray", "otherchecking", "reqmore", "answer", "kb"]
ALL_ONTOLOGY = ONTOLOGY_PAT + ONTOLOGY_DOC


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == "slot":
            for slot in sent:
                for s in slot.split(" "):
                    self.index_word(s)
        elif type == "belief":
            for slot, value in sent.items():
                for s in slot.split(" "):
                    self.index_word(s)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


# TODO: TBC
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        # self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.pat_turn_belief = data_info['pat_turn_belief']
        self.doc_turn_belief = data_info['doc_turn_belief']
        self.pat_gating_label = data_info['pat_gating_label']
        self.doc_gating_label = data_info['doc_gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.pat_generate_y = data_info["pat_generate_y"]
        self.doc_generate_y = data_info["doc_generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        pat_turn_belief = self.pat_turn_belief[index]
        doc_turn_belief = self.doc_turn_belief[index]
        pat_gating_label = self.pat_gating_label[index]
        doc_gating_label = self.doc_gating_label[index]
        turn_uttr = self.turn_uttr[index]
        pat_generate_y = self.pat_generate_y[index]
        doc_generate_y = self.doc_generate_y[index]
        pat_generate_y = self.preprocess_slot(pat_generate_y, self.trg_word2id)
        doc_generate_y = self.preprocess_slot(doc_generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            "ID": ID,
            "turn_id": turn_id,
            "pat_turn_belief": pat_turn_belief,
            "doc_turn_belief": doc_turn_belief,
            "pat_gating_label": pat_gating_label,
            "doc_gating_label": doc_gating_label,
            "context": context,
            "context_plain": context_plain,
            "turn_uttr_plain": turn_uttr,
            "pat_generate_y": pat_generate_y,
            "doc_generate_y": doc_generate_y,
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            s, v = value
            # s = s.replace("book","").strip()
            s = s.strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story


# TODO: TBC
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len - len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)  # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i, :end, :] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    pat_y_seqs, pat_y_lengths = merge_multi_response(item_info["pat_generate_y"])
    doc_y_seqs, doc_y_lengths = merge_multi_response(item_info["doc_generate_y"])
    pat_gating_label = torch.tensor(item_info["pat_gating_label"])
    doc_gating_label = torch.tensor(item_info["doc_gating_label"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        pat_gating_label = pat_gating_label.cuda()
        doc_gating_label = doc_gating_label.cuda()
        pat_y_seqs = pat_y_seqs.cuda()
        doc_y_seqs = doc_y_seqs.cuda()
        pat_y_lengths = pat_y_lengths.cuda()
        doc_y_lengths = doc_y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["pat_gating_label"] = pat_gating_label
    item_info["doc_gating_label"] = doc_gating_label
    item_info["pat_generate_y"] = pat_y_seqs
    item_info["doc_generate_y"] = doc_y_seqs
    item_info["pat_y_lengths"] = pat_y_lengths
    item_info["doc_y_lengths"] = doc_y_lengths

    return item_info


def read_langs(file_name, gating_dict, dataset, lang, mem_lang, training, max_line=None):
    print(("Reading from {}".format(file_name)))
    data = []
    # max_resp_len, max_value_len_pat, max_value_len_doc = 0, 0, 0
    max_resp_len, max_value_len = 0, 0
    pat_slot_temp = ONTOLOGY_PAT
    doc_slot_temp = ONTOLOGY_DOC

    with open(file_name) as f:
        dials = json.load(f)
        # Create vocab
        for dial_dict in dials:
            if (args["all_vocab"] or dataset == "train") and training:
                for turn_idx, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["patient_transcript"], 'utter')
                    lang.index_words(turn["doctor_transcript"], 'utter')

        # Read data
        for dial_dict in dials:
            dialog_history = ""
            for turn_idx, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]
                turn_uttr = turn["patient_transcript"] + " ; " + turn["doctor_transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += (turn["patient_transcript"] + " ; " + turn["doctor_transcript"] + " ; ")
                source_text = dialog_history.strip()

                pat_turn_belief_dict = OrderedDict([(sv[0], sv[1]) for sv in turn["pat_belief_state"]])
                doc_turn_belief_dict = OrderedDict([(sv[0], sv[1]) for sv in turn["doc_belief_state"]])
                pat_turn_belief_list = [str(k) + '-' + str(v) for k, v in pat_turn_belief_dict.items()]
                doc_turn_belief_list = [str(k) + '-' + str(v) for k, v in doc_turn_belief_dict.items()]

                if (args["all_vocab"] or dataset == "train") and training:
                    mem_lang.index_words(pat_turn_belief_dict, 'belief')
                    mem_lang.index_words(doc_turn_belief_dict, 'belief')

                pat_generate_y, doc_generate_y, pat_gating_label, doc_gating_label = [], [], [], []
                for slot in pat_slot_temp:
                    if slot in pat_turn_belief_dict.keys():
                        pat_generate_y.append(pat_turn_belief_dict[slot])
                        if pat_turn_belief_dict[slot] == "":
                            pat_gating_label.append(gating_dict["none"])
                        else:
                            pat_gating_label.append(gating_dict["ptr"])
                        if max_value_len < len(pat_turn_belief_dict[slot]):
                            max_value_len = len(pat_turn_belief_dict[slot])
                    else:
                        pat_generate_y.append("none")
                        pat_gating_label.append(gating_dict["none"])

                for slot in doc_slot_temp:
                    if slot in doc_turn_belief_dict.keys():
                        doc_generate_y.append(doc_turn_belief_dict[slot])
                        if doc_turn_belief_dict[slot] == "":
                            doc_gating_label.append(gating_dict["none"])
                        else:
                            doc_gating_label.append(gating_dict["ptr"])
                        if max_value_len < len(doc_turn_belief_dict[slot]):
                            max_value_len = len(doc_turn_belief_dict[slot])
                    else:
                        doc_generate_y.append("none")
                        doc_gating_label.append(gating_dict["none"])

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "turn_id": turn_id,
                    "dialog_history": source_text,
                    "pat_turn_belief": pat_turn_belief_list,
                    "doc_turn_belief": doc_turn_belief_list,
                    "pat_gating_label": pat_gating_label,
                    "doc_gating_label": doc_gating_label,
                    "turn_uttr": turn_uttr_strip,
                    "pat_generate_y": pat_generate_y,
                    "doc_generate_y": doc_generate_y
                }
                data.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

    # add t{} to the lang file
    if "t{}".format(max_value_len - 1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    return data, max_resp_len, pat_slot_temp, doc_slot_temp


# TODO: TBC
def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    os.environ["HOME"] = "D:/ANAHOME"       # add HOME directory temporarily
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


# def dump_torchtext_emb(word2index, index2word, dump_path):
#     print("Dumping pretrained embeddings from torchtext...")


def prepare_data_seq(training, sequicity=0, batch_size=64):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'
    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        folder_name = 'save/{}-'.format(args["decoder"]) + args["addName"] + args['dataset'] + str(args['task']) + '/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ALL_SLOTS = ALL_ONTOLOGY
    gating_dict = {"ptr": 0, "none": 1}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, pat_slot_train, doc_slot_train = read_langs(file_train, gating_dict, "train", lang,
                                                                               mem_lang, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, pat_slot_dev, doc_slot_dev = read_langs(file_dev, gating_dict, "dev", lang,
                                                                       mem_lang, training)
        dev = get_seq(pair_dev, lang, mem_lang, batch_size, True, sequicity)
        pair_test, test_max_len, pat_slot_test, doc_slot_test = read_langs(file_test, gating_dict, "test", lang,
                                                                           mem_lang, training)
        test = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle:
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle:
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle:
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle:
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)
        pair_train, train_max_len, pat_slot_train, doc_slot_train, train, nb_train_vocab = [], 0, {}, {}, [], 0
        pair_dev, dev_max_len, pat_slot_dev, doc_slot_dev = read_langs(file_dev, gating_dict, "dev", lang,
                                                                       mem_lang, training)
        dev = get_seq(pair_dev, lang, mem_lang, batch_size, True, sequicity)
        pair_test, test_max_len, pat_slot_test, doc_slot_test = read_langs(file_test, gating_dict, "test", lang,
                                                                           mem_lang, training)
        test = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab)
    print("Vocab_size Belief %s" % mem_lang.n_words)
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, pat_slot_train, doc_slot_train, pat_slot_dev, doc_slot_dev, pat_slot_test, doc_slot_test]
    LANG = [lang, mem_lang]
    return train, dev, test, LANG, SLOTS_LIST, gating_dict, nb_train_vocab
