import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

import os
import json
import copy

from utils.masked_cross_entropy import *
from utils.config import *


class CDST(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict):
        super(CDST, self).__init__()
        self.name = "CDST"
        self.task = task
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        # TODO: TBC
        self.pat_slot_temp = slots[3]
        self.doc_slot_temp = slots[4]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th')
                trained_decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)

            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

            # Initialize optimizers and criterion
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                            min_lr=0.0001, verbose=True)

            self.reset()
            if USE_CUDA:
                self.encoder.cuda()
                self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        directory = 'save/CDST-' + args["addName"] + args['dataset'] + str(self.task) + '/' + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, pat_slot_temp, doc_slot_temp, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        # TODO: determine the input and output of the model
        pat_gen_out, doc_gen_out, pat_gates, doc_gates, pat_words_point_out, doc_words_point_out, pat_words_class_out, \
        doc_words_class_out = self.joint_encode_and_decode(data, use_teacher_forcing, pat_slot_temp, doc_slot_temp)

        pat_loss_ptr = masked_cross_entropy_for_value(
            pat_gen_out.transpose(0, 1).contiguous(),
            data["pat_generate_y"].contiguous(),
            data["pat_y_lengths"])
        ptr_loss_gate = self.cross_entorpy(pat_gates.transpose(0, 1).contiguous().view(-1, pat_gates.size(-1)),
                                           data["gating_label"].contiguous().view(-1))

        doc_loss_ptr = masked_cross_entropy_for_value(
            doc_gen_out.transpose(0, 1).contiguous(),
            data["doc_generate_y"].contiguous(),
            data["doc_y_lengths"])
        doc_loss_gate = self.cross_entorpy(doc_gates.transpose(0, 1).contiguous().view(-1, doc_gates.size(-1)),
                                           data["gating_label"].contiguous().view(-1))

        total_loss = pat_loss_ptr + ptr_loss_gate + doc_loss_ptr + doc_loss_gate
        loss_ptr = pat_loss_ptr + doc_loss_ptr
        loss_gate = ptr_loss_gate + doc_loss_gate

        self.loss_grad = total_loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += total_loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def joint_encode_and_decode(self, data, use_teacher_forcing, pat_slot_temp, doc_slot_temp):
        # Build unknown mask for memory to encourage generalization
        story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        pat_max_res_len = data['pat_generate_y'].size(2) if self.encoder.training else 10
        doc_max_res_len = data['doc_generate_y'].size(2) if self.encoder.training else 10
        pat_gen_out, doc_gen_out, pat_gates, doc_gates, pat_words_point_out, doc_words_point_out, \
        pat_words_class_out, doc_words_class_out = self.decoder.forward(batch_size, encoded_hidden, encoded_outputs,
                                                                        data['context_len'], story, pat_max_res_len,
                                                                        doc_max_res_len, data['pat_generate_y'],
                                                                        data['doc_generate_y'], use_teacher_forcing,
                                                                        pat_slot_temp, doc_slot_temp)

        return pat_gen_out, doc_gen_out, pat_gates, doc_gates, pat_words_point_out, doc_words_point_out, \
               pat_words_class_out, doc_words_class_out

    def evaluate(self, dev, metric_best, pat_slot_temp, doc_slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, _, pat_gates, doc_gates, pat_words, doc_words, pat_class_words, doc_class_words = \
                self.joint_encode_and_decode(data_dev, False, pat_slot_temp, doc_slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                 [data_dev["ID"][bi]][data_dev["turn_id"][bi]] = \
                    {"pat_turn_belief": data_dev["pat_turn_belief"][bi],
                     "doc_turn_belief": data_dev["doc_turn_belief"][bi]}

                pat_gate = torch.argmax(pat_gates.transpose(0, 1)[bi], dim=1)
                doc_gate = torch.argmax(doc_gates.transpose(0, 1)[bi], dim=1)
                pat_predict_belief, doc_predict_belief = [], []

                # pointer-generator results
                pat_predict_belief = self.get_ptr_gen(bi, pat_gate, pat_words, pat_slot_temp)
                doc_predict_belief = self.get_ptr_gen(bi, doc_gate, doc_words, doc_slot_temp)

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pat_pred"] = pat_predict_belief
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["doc_pred"] = doc_predict_belief

                if set(data_dev["pat_turn_belief"][bi]) != set(pat_predict_belief) and args["genSample"]:
                    print("True", set(data_dev["pat_turn_belief"][bi]))
                    print("Pred", set(pat_predict_belief), "\n")
                if set(data_dev["doc_turn_belief"][bi]) != set(doc_predict_belief) and args["genSample"]:
                    print("True", set(data_dev["doc_turn_belief"][bi]))
                    print("Pred", set(doc_predict_belief), "\n")

                if args["genSample"]:
                    json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

                pat_joint_acc, pat_F1, pat_turn_acc = self.evaluate_metrics(all_prediction, "pat_turn_belief",
                                                                            "pat_pred", pat_slot_temp)
                doc_joint_acc, doc_F1, doc_turn_acc = self.evaluate_metrics(all_prediction, "doc_turn_belief",
                                                                            "doc_pred", doc_slot_temp)

                pat_evaluation_metrics = {"Joint Acc":pat_joint_acc, "Turn Acc":pat_turn_acc, "Joint F1":pat_F1}
                doc_evaluation_metrics = {"Joint Acc":doc_joint_acc, "Turn Acc":doc_turn_acc, "Joint F1":doc_F1}
                print("Patient: ", pat_evaluation_metrics)
                print("Doctor: ", doc_evaluation_metrics)
                joint_acc = (pat_joint_acc + doc_joint_acc) / 2
                F1 = (pat_F1 + doc_F1) / 2

                if early_stop == 'F1':
                    if F1 >= metric_best:
                        self.save_model('ENTF1-{:.4f}'.format(F1))
                        print("MODEL SAVED")
                    return F1
                else:
                    if joint_acc >= metric_best:
                        self.save_model('ACC-{:.4f}'.format(joint_acc))
                        print("MODEL SAVED")
                    return joint_acc

    def get_ptr_gen(self, bi, gate, words, slot_temp):
        predict_belief = []
        for si, sg in enumerate(gate):
            if sg == self.gating_dict["none"]:
                continue
            elif sg == self.gating_dict["ptr"]:
                pred = np.transpose(words[si])[bi]
                st = []
                for e in pred:
                    if e == 'EOS':
                        break
                    else:
                        st.append(e)
                st = " ".join(st)
                if st == "none":
                    continue
                else:
                    predict_belief.append(slot_temp[si] + "-" + str(st))

        return predict_belief

    def evaluate_metrics(self, all_prediction, turn_belief, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv[turn_belief]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv[turn_belief]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv[turn_belief]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)



class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}


    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


