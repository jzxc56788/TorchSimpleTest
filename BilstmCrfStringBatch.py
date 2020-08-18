# -*- coding: utf-8 -*-

import sys
import time

import torch
import torch.nn as nn
import torch.jit as Tensor
from typing import List, Tuple, Dict, AnyStr
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return int(idx.item())



class BilstmCrfStringBatch(torch.nn.Module):
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, vocab, device = "cpu"):
        super(BilstmCrfStringBatch, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device
        self.batch_size = 1

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device = self.device), torch.randn(2, self.batch_size, self.hidden_dim // 2, device = self.device))

    # batch 矩陣運算版 _forward_alg
    def _forward_alg_m_b(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device = self.device)
        init_alphas[0][self.tag_to_ix["<START>"]] = 0.
        forward_var = init_alphas.expand(self.batch_size, self.tagset_size)
        for feat in feats:
            batch_forward_var_m = torch.tensor([], device = self.device)
            batch_emit_score_m = torch.tensor([], device = self.device)
            # batch_transitions = torch.tensor([], device = self.device)
            batch_transitions = self.transitions.t().view(1, -1, self.tagset_size).repeat(self.batch_size, 1, 1)
            for b_i in range(len(feat)):
                forward_var_m = forward_var[b_i].expand(self.tagset_size, self.tagset_size).view(1, -1, self.tagset_size)
                batch_forward_var_m = torch.cat((batch_forward_var_m, forward_var_m), 0)

                emit_score_m = feat[b_i].expand(self.tagset_size, self.tagset_size).t().view(1, -1, self.tagset_size)
                batch_emit_score_m = torch.cat((batch_emit_score_m, emit_score_m), 0)

                # batch_transitions = torch.cat((batch_transitions, self.transitions.t().view(1, -1, self.tagset_size)), 0)

            next_tag_var_m = batch_emit_score_m + batch_forward_var_m + batch_transitions
            forward_var = self._batch_log_sum_exp(next_tag_var_m)

        terminal_var_m = forward_var + self.transitions[self.tag_to_ix["<STOP>"]].expand(forward_var.size())
        max_score_b = torch.max(terminal_var_m, 1).values

        max_score_broadcast_m = max_score_b.view(1, -1).t().expand(terminal_var_m.size())
        alpha_b = max_score_b + torch.log(torch.sum(torch.exp(terminal_var_m - max_score_broadcast_m), 1))
        return alpha_b

    def _batch_log_sum_exp(self, vec):
        max_score_batch = torch.max(vec, 2).values
        max_score_broadcast_m_batch = torch.tensor([], device = self.device)
        for max_score in max_score_batch:
            max_score_m = max_score.expand(self.tagset_size, self.tagset_size).t().view(1, -1, self.tagset_size)
            max_score_broadcast_m_batch = torch.cat((max_score_broadcast_m_batch, max_score_m), 0)
        return max_score_batch + torch.log(torch.sum(torch.exp(vec - max_score_broadcast_m_batch), 2))

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(self.batch_size, device = self.device)
        batch_start_tag = torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long, device = self.device).expand(self.batch_size, -1)
        # tags = torch.cat([torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long, device = self.device), tags[0]])
        tags = torch.cat((batch_start_tag, tags), 1)
        for i, feat_b in enumerate(feats):
            for b_i, feat in enumerate(feat_b):
                score[b_i] = score[b_i] + self.transitions[tags[b_i, i + 1], tags[b_i, i]] + feat[tags[b_i, i + 1]]
            # score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[:, -1]]
        
        # print(self.transitions[self.tag_to_ix["<STOP>"], tags[:, -1]])
        return score

    def neg_log_likelihood(self, sentence, tags):
        seconds = time.time()
        feats = self._get_lstm_features(sentence)
        # print('LSTM : ', time.time() - seconds,'s')
        seconds = time.time()

        forward_score = self._forward_alg_m_b(feats)
        # print('CRFF : ', time.time() - seconds,'s')
        seconds = time.time()

        gold_score = self._score_sentence(feats, tags)
        # print('SCOR : ', time.time() - seconds,'s')
        seconds = time.time()
        # print("score: ", gold_score, "| total: ", forward_score, "| max score: ", torch.max(forward_score - gold_score))
        # print("max score: ", torch.max(forward_score - gold_score))
        return torch.sum(forward_score - gold_score) / self.batch_size

    def _get_lstm_features(self, sentence):
        if len(sentence.size()) == 2:
            self.batch_size = sentence.size()[0]
            input_size = sentence.size()[1]
            self.hidden = self.init_hidden()
        else:
            self.batch_size = 1
            input_size = sentence.size()[0]
            self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(input_size, self.batch_size, -1)
        lstm_out, _ = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(input_size, self.batch_size, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _viterbi_decode(self, feats):
        backpointers: List[List[int]] = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t: List[int] = []  # holds the backpointers for this step
            viterbivars_t: List[torch.Tensor] = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id: int = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][int(best_tag_id)].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][int(best_tag_id)]

        best_path: List[int] = [best_tag_id]
        for i in range(len(backpointers) - 1, -1, -1):
            best_tag_id = backpointers[i][best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix["<START>"]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def _forward_lstm_features(self, sentence):
        self.batch_size = 1
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, input):
        sentence = self.encode(input)
        # print(f'sentence:{sentence}')
        lstm_feats = self._forward_lstm_features(sentence)
        # print(f'lstm_feats:{lstm_feats}')
        score, tag_seq = self._viterbi_decode(lstm_feats)
        # print(f'score:{score}')
        # print(f'tag_seq:{tag_seq}')
        id2tag_output = self.id2tag(tag_seq)
        # print(f'id2tag_output:{id2tag_output}')
        # quit()
        return self.decode(id2tag_output, input)
    
    def id2tag(self, predict_tag_seq: List[int]):
        id2tag_output: List[str] = []
        for tag_id in predict_tag_seq:
            for tag, id in self.tag_to_ix.items():
                if tag_id == id:
                    id2tag_output.append(tag)
                    break
        return id2tag_output

    def decode(self, tags: List[str], input):
        tag_class: Dict[str, List[str]] = {}
        last_tag = ""
        print(tags)
        for tag, w in zip(tags, input):
            w = str(w)
            if tag == 'O':
                if not tag in tag_class.keys():
                    tag_class[tag] = [w]
                else:
                    if last_tag != tag:
                        tag_class[tag].append(w)
                    else:
                        tag_class[tag][len(tag_class[tag])-1] += w
                last_tag = tag
            else:
                s_tag = tag.split('-')
                bi_tag = s_tag[0]
                big_tag = s_tag[1]
                if not big_tag in tag_class.keys():
                    tag_class[big_tag] = [w]
                elif bi_tag == 'B':
                    tag_class[big_tag].append(w)
                else:
                    if last_tag != big_tag:
                        tag_class[big_tag].append(w)
                    else:
                        tag_class[big_tag][len(tag_class[big_tag])-1] += w
                last_tag = big_tag
        return tag_class
        
    def encode(self, predict_text):
        w2id = []
        for w in predict_text:
            has_vocab = False
            for i in range(len(self.vocab)):
                if self.vocab[i] == str(w) :
                    has_vocab = True
                    w2id.append(i)
                    break
            if not has_vocab:
                w2id.append(-1)
        w2id = torch.tensor(w2id).to(torch.int64)
        return w2id
