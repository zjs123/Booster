import heapq
from pathlib import Path
import pickle
from typing import Dict, Tuple, List
from Utils import dtw_measure
import faulthandler

import numpy as np
import torch
import random
from models import TKBCModel
from tqdm import tqdm
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


DATA_PATH = 'data/'

faulthandler.enable()
class DisDataset(Dataset):
    def __init__(self, dis_data, name, Type='train'):
        self.Type = Type

        self.root = Path(DATA_PATH) / name
        self.name = name
        in_file = open(str(self.root / ('train' + '.pickle')), 'rb')

        self.id_2_t = self.read_dict(str(self.root / f'ts_id'))
        try:
            self.id_2_e = self.read_dict(str(self.root / f'ent_id'))
            self.id_2_r = self.read_dict(str(self.root / f'rel_id'))
            self.n_entity = len(self.id_2_e)
            self.n_rel = len(self.id_2_r)*2
        except:
            if 'yago' in name:
                self.n_entity = 10623
                self.n_rel = 10*2
            if 'wiki' in name:
                self.n_entity = 12554
                self.n_rel = 24*2
        self.n_timestamps = len(self.id_2_t)
        
        train_data = pickle.load(in_file)
        copy = np.copy(train_data)
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_rel // 2  # has been multiplied by two.
        self.train_data = np.vstack((train_data, copy))
        self.train_data_raw = train_data
        
        self.aux_dict, self.aux_dict_ere, self.ee2rt, self.er2et, self.ee2rt_num, self.er2et_num, self.r2pairs, self.pair2tr, self.r2s, self.e2te, self.e2tr, self.e2e = self.get_aux_dict()

        self.noise_num = 3
        self.time_window = 50
        self.all_data = []
        self.data_num = []
        for sample in dis_data:
            if len(sample) == 1:
                continue
            self.all_data.append(sample)
            self.data_num.append(len(sample))
        self.predict_data = self.flatten_list()

        self.rel_score_dict = self.get_rel_score()
        self.ent_score_dict = self.get_ent_score()

        self.history_save_dict = {}
        self.score_dict = {}
    
    def get_score_dict(self):
        return self.score_dict
    
    def read_dict(self, path):
        id_2_str = {}
        dict_file = open(path)
        for line in dict_file.readlines():
            str_name = line.strip().split('\t')[0]
            if 'ts' in path:
                str_name = str_name.replace('-', ' ')
            str_id = int(line.strip().split('\t')[1])
            id_2_str[str_id] = str_name
        return id_2_str
    
    def get_aux_dict(self):
        e2e_rt_dict = {}
        e2r_et_dict = {}
        ee2rt_dict = {}
        er2et_dict = {}
        ee2rt_num_dict = {}
        er2et_num_dict = {}
        r2pair = {}
        pair2t2r = {}
        e2t2e_dict = {}
        e2t2r_dict = {}
        r2s = {}
        e2e = {}
        for sample in self.train_data:
            h_e = int(sample[0])
            r = int(sample[1])
            t_e = int(sample[2])
            t = int(sample[3])

            pair_index = str([h_e, t_e])

            # get e2e dict
            if h_e not in e2e.keys():
                e2e[h_e] = set()
            e2e[h_e].add(t_e)

            if t_e not in e2e.keys():
                e2e[t_e] = set()
            e2e[t_e].add(h_e)


            # get r2s
            if r not in r2s.keys():
                r2s[r] = set()
            r2s[r].add(h_e)

            # get r2pair
            if r not in r2pair.keys():
                r2pair[r] = {}
                r2pair[r][pair_index] = []
            else:
                if pair_index not in r2pair[r].keys():
                    r2pair[r][pair_index] = []
            r2pair[r][pair_index].append(t)

            # get pair2t2r
            if pair_index not in pair2t2r.keys():
                pair2t2r[pair_index] = {}
                pair2t2r[pair_index][t] = set()
            else:
                if t not in pair2t2r[pair_index].keys():
                    pair2t2r[pair_index][t] = set()
            pair2t2r[pair_index][t].add(r)

            # get ee2rt_dict
            if h_e not in ee2rt_dict.keys():
                ee2rt_dict[h_e] = {}
                ee2rt_dict[h_e][t_e] = [[], []]
                ee2rt_dict[h_e][t_e][0].append(r)
                ee2rt_dict[h_e][t_e][1].append(t)
            else:
                if t_e not in ee2rt_dict[h_e].keys():
                    ee2rt_dict[h_e][t_e] = [[], []]
                    ee2rt_dict[h_e][t_e][0].append(r)
                    ee2rt_dict[h_e][t_e][1].append(t)
                else:
                    ee2rt_dict[h_e][t_e][0].append(r)
                    ee2rt_dict[h_e][t_e][1].append(t)
            
            # ee2rt_num_dict
            if h_e not in ee2rt_num_dict.keys():
                ee2rt_num_dict[h_e] = {}
                ee2rt_num_dict[h_e][t_e] = 0
            else:
                if t_e not in ee2rt_num_dict[h_e].keys():
                    ee2rt_num_dict[h_e][t_e] = 0
            ee2rt_num_dict[h_e][t_e] += 1
            
            # get er2et_dict
            if h_e not in er2et_dict.keys():
                er2et_dict[h_e] = {}
                er2et_dict[h_e][r] = [[], []]
                er2et_dict[h_e][r][0].append(t_e)
                er2et_dict[h_e][r][1].append(t)
            else:
                if r not in er2et_dict[h_e].keys():
                    er2et_dict[h_e][r] = [[], []]
                    er2et_dict[h_e][r][0].append(t_e)
                    er2et_dict[h_e][r][1].append(t)
                else:
                    er2et_dict[h_e][r][0].append(t_e)
                    er2et_dict[h_e][r][1].append(t)
            
            # get er2et_num_dict
            if h_e not in er2et_num_dict.keys():
                er2et_num_dict[h_e] = {}
                er2et_num_dict[h_e][r] = 0
            else:
                if r not in er2et_num_dict[h_e].keys():
                    er2et_num_dict[h_e][r] = 0
            er2et_num_dict[h_e][r] += 1

            # get e_r_e_t
            if h_e not in e2r_et_dict.keys():
                e2r_et_dict[h_e] = {}
                e2r_et_dict[h_e][r] = {}
                e2r_et_dict[h_e][r][t_e] = set()
                e2r_et_dict[h_e][r][t_e].add(t)
            else:
                if r not in e2r_et_dict[h_e].keys():
                    e2r_et_dict[h_e][r] = {}
                    e2r_et_dict[h_e][r][t_e] = set()
                    e2r_et_dict[h_e][r][t_e].add(t)
                else:
                    if t_e not in e2r_et_dict[h_e][r].keys():
                        e2r_et_dict[h_e][r][t_e] = set()
                        e2r_et_dict[h_e][r][t_e].add(t)
                    else:
                        e2r_et_dict[h_e][r][t_e].add(t)
            
            # get e_e_r_t
            if h_e not in e2e_rt_dict.keys():
                e2e_rt_dict[h_e] = {}
                e2e_rt_dict[h_e][t_e] = {}
                e2e_rt_dict[h_e][t_e][r] = set()
                e2e_rt_dict[h_e][t_e][r].add(t)
            else:
                if t_e not in e2e_rt_dict[h_e].keys():
                    e2e_rt_dict[h_e][t_e] = {}
                    e2e_rt_dict[h_e][t_e][r] = set()
                    e2e_rt_dict[h_e][t_e][r].add(t)
                else:
                    if r not in e2e_rt_dict[h_e][t_e].keys():
                        e2e_rt_dict[h_e][t_e][r] = set()
                        e2e_rt_dict[h_e][t_e][r].add(t)
                    else:
                        e2e_rt_dict[h_e][t_e][r].add(t)
            
            # get e2t2e dict
            if h_e not in e2t2e_dict.keys():
                e2t2e_dict[h_e] = {}
                e2t2e_dict[h_e][t] = set()
            else:
                if t not in e2t2e_dict[h_e].keys():
                    e2t2e_dict[h_e][t] = set()
            e2t2e_dict[h_e][t].add(t_e)

            # get e2t2r dict
            if h_e not in e2t2r_dict.keys():
                e2t2r_dict[h_e] = {}
                e2t2r_dict[h_e][t] = set()
            else:
                if t not in e2t2r_dict[h_e].keys():
                    e2t2r_dict[h_e][t] = set()
            e2t2r_dict[h_e][t].add(r)
        
        # sorted e2t2e dict
        e2te = {}
        for s in e2t2e_dict.keys():
            raw_t2e = e2t2e_dict[s]
            e2te[s] = []
            for time in range(self.n_timestamps):
                if time in raw_t2e.keys():
                    e2te[s].append(raw_t2e[time])
                else:
                    e2te[s].append(set([]))
        
        # sorted e2t2r dict
        e2tr = {}
        for s in e2t2r_dict.keys():
            raw_t2r = e2t2r_dict[s]
            e2tr[s] = []
            for time in range(self.n_timestamps):
                if time in raw_t2r.keys():
                    e2tr[s].append(raw_t2r[time])
                else:
                    e2tr[s].append(set([]))
        
        # sorted pair2t2r dict
        pair2tr = {}
        for pair in pair2t2r.keys():
            raw_t2r = pair2t2r[pair]
            pair2tr[pair] = []
            for time in range(self.n_timestamps):
                if time in raw_t2r.keys():
                    pair2tr[pair].append(raw_t2r[time])
                else:
                    pair2tr[pair].append(set([]))

        return e2e_rt_dict, e2r_et_dict, ee2rt_dict, er2et_dict, ee2rt_num_dict, er2et_num_dict, r2pair, pair2tr, r2s, e2te, e2tr, e2e

    
    def get_rel_score(self): # (r_1, r_2, ?) relational triadic by entity anonymization
        tmp_rel_score = {}

        for r in tqdm(self.r2pairs.keys()):
            if r not in tmp_rel_score.keys():
                tmp_rel_score[r] = {}
            pair_list = self.r2pairs[r].keys()
            for pair in pair_list:
                e_1, e_2 = eval(pair)
                t_list = self.r2pairs[r][pair]
                for t in t_list:
                    e1_nei_set = set()
                    for tmp_set in self.e2te[e_1][max(0, t-2):min(self.n_timestamps-1, t+2)]:
                        e1_nei_set = e1_nei_set | tmp_set
                    e2_nei_set = set() 
                    for tmp_set in self.e2te[e_2][max(0, t-2):min(self.n_timestamps-1, t+2)]:
                        e2_nei_set = e2_nei_set | tmp_set
                    interact_e = e1_nei_set&e2_nei_set
                    
                    tmp_pair_set = set()
                    for e in interact_e:
                        #tmp_pair_set = set()
                        r1_nei_set = set()
                        for tmp_set in self.pair2tr[str(sorted([e, e_1]))][max(0, t-2):min(self.n_timestamps-1, t+2)]:
                            r1_nei_set = r1_nei_set | tmp_set
                        r2_nei_set = set()
                        for tmp_set in self.pair2tr[str(sorted([e, e_2]))][max(0, t-2):min(self.n_timestamps-1, t+2)]:
                            r2_nei_set = r2_nei_set | tmp_set
                        
                        for r1 in r1_nei_set:
                            for r2 in r2_nei_set:
                                r_pair_index = str(sorted([r1, r2]))
                                tmp_pair_set.add(r_pair_index)
                                
                                if r_pair_index not in tmp_rel_score[r].keys():
                                    tmp_rel_score[r][r_pair_index] = set()
                                tmp_rel_score[r][r_pair_index].add(str(sorted([e,e_1,e_2])))
                    
        rel_score = {}
        max_num = 0
        min_num = 1000              
        for key_1 in tmp_rel_score.keys():
            for key_2 in tmp_rel_score[key_1].keys():
                r1 = key_1
                r2, r3 = eval(key_2)
                try:
                    tmp_set_1 = tmp_rel_score[r1][str(sorted([r2,r3]))]
                except:
                    tmp_set_1 = set()
                score_1 = len(tmp_set_1)
                if max_num < score_1:
                    max_num = score_1
                if min_num > score_1:
                    min_num = score_1
        
        range_ = max_num - min_num
        rel_score = {}
        for key_1 in tmp_rel_score.keys():
            rel_score[key_1] = {}
            for key_2 in tmp_rel_score[key_1].keys():
                r1 = key_1
                r2, r3 = eval(key_2)
                try:
                    tmp_set_1 = tmp_rel_score[r1][str(sorted([r2,r3]))]
                except:
                    tmp_set_1 = set()
                score_1 = len(tmp_set_1)
                rel_score[key_1][key_2] = float(score_1-min_num)/float(range_)
        
        return rel_score

    def get_ent_score(self): # (e_1, e_2, e_3) relational triadic by relation anonymization, e_2, e_3 co-occur in the same e_x
        tmp_ent_score = {}
        for e_1 in tqdm(self.ee2rt.keys()):
            for e_2 in self.ee2rt[e_1].keys():
                pair_index = str(sorted([e_1, e_2]))
                if pair_index not in tmp_ent_score.keys():
                    tmp_ent_score[pair_index] = {}
                e1_nei_set = self.e2e[e_1]
                e2_nei_set = self.e2e[e_2]       
                interact_e = e1_nei_set&e2_nei_set
                for e in interact_e:
                    e_nei_set = self.e2e[e]   
                    interact_e1 = e1_nei_set&e_nei_set
                    interact_e2 = e2_nei_set&e_nei_set
                    if e not in tmp_ent_score[pair_index].keys():
                        tmp_ent_score[pair_index][e] = 0
                    tmp_ent_score[pair_index][e] = np.min([len(interact_e), len(interact_e1), len(interact_e2)])
        
        max_num = 0
        min_num = 1000              
        for key_1 in tmp_ent_score.keys():
            for key_2 in tmp_ent_score[key_1].keys():
                e1, e2 = eval(key_1)
                e3 = key_2
                try:
                    score_1 = tmp_ent_score[str(sorted([e1,e2]))][e3]
                except:
                    continue
                if max_num < score_1:
                    max_num = score_1
                if min_num > score_1:
                    min_num = score_1
        
        range_ = max_num - min_num
        ent_score = {}
        for key_1 in tmp_ent_score.keys():
            ent_score[key_1] = {}
            for key_2 in tmp_ent_score[key_1].keys():
                e1, e2 = eval(key_1)
                e3 = key_2
                try:
                    score_1 = tmp_ent_score[str(sorted([e1,e2]))][e3]
                except:
                    continue
                ent_score[key_1][key_2] = float(score_1-min_num)/float(range_)
            return ent_score
    
    def flatten_list(self):
        flatten_list = []
        for data in self.all_data:
            index = 1
            while index < len(data):
                flatten_list.append([data[0], data[index]])
                index += 1
        return flatten_list
    
    def np_softmax(self, f):
        
        f_max = np.max(f)
        f_exp = np.exp(f-f_max)
        f_exp = f_exp * (f != 0)
        Sum = np.sum(f_exp)
        Sum = Sum + (Sum == 0.0)+0.1
        score = f_exp / Sum
        '''
        f -= np.max(f)
        score = np.exp(f)/np.sum(np.exp(f))
        '''

        return score
    
    def np_sigmoid(self, f):
        z = 1/(1 + np.exp(-f))
        return z
    
    def select_top_ent(self, e_list, anchor_e, query_e, query_t, top_K):
        score_list = []
        t_list = []
        sorted_anchor_query_index = str(sorted([anchor_e, query_e]))
        for e in e_list:
            try:
                score_list.append(self.ent_score_dict[sorted_anchor_query_index][e])
            except:
                score_list.append(0)
            try:
                tmp_t_a = np.min(np.abs(np.array(self.ee2rt[anchor_e][e][1]) - query_t))
                tmp_t_q = np.min(np.abs(np.array(self.ee2rt[query_e][e][1]) - query_t))
                t_list.append(np.mean([tmp_t_a, tmp_t_q]))
            except:
                t_list.append(-1000)
        
        tmp_score_list = self.np_softmax(score_list)+self.np_softmax(t_list)
        top_k_index = np.array(score_list).argsort()[-top_K:]
        return np.array(e_list)[top_k_index]
    
    def get_history(self, anchor_ent, query_rel, query_ent, query_time, type):
        if 'ICEWS' in self.name:
            time_window = 50
        else:
            time_window = 1
        t_score_list = []
        ent_score_list = []
        rel_score_list = []
        
        # get co_neighbor list
        #anchor_nei_set = e2e[anchor_ent] 
        anchor_nei_set = set()
        for tmp_set in self.e2te[anchor_ent][max(0, query_time-time_window):min(self.n_timestamps-1, query_time+time_window)]:
            anchor_nei_set = anchor_nei_set|tmp_set
        #query_nei_set = e2e[query_ent] 
        query_nei_set = set()
        for tmp_set in self.e2te[query_ent][max(0, query_time-time_window):min(self.n_timestamps-1, query_time+time_window)]:
            query_nei_set = query_nei_set|tmp_set
        key_ent_list = list(anchor_nei_set&query_nei_set)
        if len(key_ent_list) >= 10:
            key_ent_list = self.select_top_ent(key_ent_list, anchor_ent, query_ent, query_time, 10)
        
        # get structured scores
        if len(key_ent_list) == 0:
            return np.array([-1]), np.array([-1]), np.array([-1])
        else:
            for key_ent in key_ent_list:
                # ent modeling
                try:
                    score_1 = self.ent_score_dict[str(sorted([anchor_ent, query_ent]))][key_ent]
                except:
                    score_1 = 0
                try:
                    score_2 = self.ent_score_dict[str(sorted([anchor_ent, key_ent]))][query_ent]
                except:
                    score_2 = 0
                try:
                    score_3 = self.ent_score_dict[str(sorted([query_ent, key_ent]))][anchor_ent]
                except:
                    score_3 = 0
                ent_score_list.append(1+np.max([score_1, score_2, score_3]))
                
                # rel modeling
                rt_list_anchor_key = self.ee2rt[anchor_ent][key_ent]
                rt_list_query_key = self.ee2rt[query_ent][key_ent]

                tmp_rel_score_list = []
                tmp_time_score_list = []
                tmp_t1_list = []
                tmp_t2_list = []
                for index_1 in range(len(rt_list_anchor_key[0])):
                    r1 = rt_list_anchor_key[0][index_1]
                    t1 = rt_list_anchor_key[1][index_1]
                    if abs(t1-query_time) > time_window//2:
                        continue
                    for index_2 in range(len(rt_list_query_key[0])):
                        r2 = rt_list_query_key[0][index_2]
                        t2 = rt_list_query_key[1][index_2]
                        if abs(t2-query_time) > time_window//2:
                            continue
                        tmp_t1_list.append(-abs(t1-query_time))
                        tmp_t2_list.append(-abs(t2-query_time))
                        
                        sorted_rel_index = str(sorted([r1, r2]))
                        try:
                            if sorted_rel_index in self.rel_score_dict[query_rel].keys():
                                tmp_rel_score_list.append(self.rel_score_dict[query_rel][sorted_rel_index])
                                tmp_time_score_list.append(-abs(t1-t2))
                        except:
                            tmp_rel_score_list.append(0)
                            tmp_time_score_list.append(-1000)

                if len(tmp_time_score_list) != 0:
                    tmp_list = self.np_softmax(tmp_time_score_list)*tmp_rel_score_list
                    max_index = np.argmax(tmp_list)
                    rel_score_list.append(np.sum(tmp_list))
                    t_score_list.append(tmp_time_score_list[max_index])
                else:
                    rel_score_list.append(0)
                    t_score_list.append(-1000)
        
        '''
        if type == 'noise':
            noise_type = random.random()
            if noise_type < 0.3:
                indices = np.random.choice(np.arange(t_score_list.size), replace=False,
                                size=1)
                t_score_list += t_score_list[indices]
                ent_score_list += ent_score_list[indices]
                rel_score_list += rel_score_list[indices]
            elif noise_type < 0.6:
                t_score_list = np.array(t_score_list)
                ent_score_list = np.array(ent_score_list)
                rel_score_list = np.array(rel_score_list)
                indices = np.random.choice(np.arange(t_score_list.size), replace=False,
                                size=1)
                t_score_list[indices] = 0
                ent_score_list[indices] = 0
                rel_score_list[indices] = 0
            elif noise_type < 0.9:
                t_score_list = np.array(t_score_list)
                ent_score_list = np.array(ent_score_list)
                rel_score_list = np.array(rel_score_list)
                indices = np.random.choice(np.arange(t_score_list.size), replace=False,
                                size=1)
                t_score_list[indices] += 0.1
                ent_score_list[indices] += 0.1
                rel_score_list[indices] += 0.1
        '''
        t_score_list = np.array(t_score_list)
        ent_score_list = np.array(ent_score_list)
        rel_score_list = np.array(rel_score_list)

        return t_score_list, ent_score_list, rel_score_list
    
    def add_noise(self, e_list, t_list):
        return e_list, t_list
        # 1.random dropout 2. time move 3. time exchange 4. random repeat 5. leaf replace
        e_list_with_noise = []
        t_list_with_noise = []
        
        if len(t_list) == 1:
            rand_singal = random.random()
            if rand_singal <= 0.5:
                # time move
                e_list_with_noise = e_list
                t_list_with_noise = [t_list[0]+random.randint(-3,3)]
            else:
                # random repeat
                rand_repeat = random.randint(2,4)
                e_list_with_noise = [e_list[0] for i in range(rand_repeat)]
                t_list_with_noise = [t_list[0]+i for i in range(rand_repeat)]
            return e_list_with_noise, t_list_with_noise
        
        else:
            rand_singal = random.random()
            if rand_singal <= 0.1:
                # time move
                e_list_with_noise = e_list
                t_list_with_noise = [t_list[i]+random.randint(-3,3) for i in range(len(t_list))]
            elif rand_singal <= 0.2:
                # random repeat
                rand_repeat = random.choices([i for i in range(len(t_list))], k=len(t_list)+4)
                e_list_with_noise = [e_list[i] for i in rand_repeat]
                t_list_with_noise = [t_list[i] for i in rand_repeat]
            elif rand_singal <= 0.3:
                # time exchange
                e_list_with_noise = e_list
                t_list_with_noise = t_list
                t_list_with_noise = random.shuffle(t_list_with_noise)
            elif rand_singal <= 0.4:
                # random dropout
                rand_repeat = random.sample([i for i in range(len(t_list))], max(0, len(t_list)-2))
                e_list_with_noise = [e_list[i] for i in rand_repeat]
                t_list_with_noise = [t_list[i] for i in rand_repeat]
            else:
                e_list_with_noise = e_list
                t_list_with_noise = t_list
        
            return e_list_with_noise, t_list_with_noise

    
    def __len__(self):
        return len(self.predict_data)
    
    def __getitem__(self, item):
        can_head = self.predict_data[item][1][0]
        can_rel = self.predict_data[item][1][1]
        can_tail = self.predict_data[item][1][2]
        can_time = self.predict_data[item][1][3]

        pos_head = self.predict_data[item][0][0]
        pos_rel = self.predict_data[item][0][1]
        pos_tail = self.predict_data[item][0][2]
        pos_time = self.predict_data[item][0][3]

        label = self.predict_data[item][1][4]
        sample_type = self.predict_data[item][1][-1]

        can_t_score_list, can_ent_score_list, can_rel_score_list = self.get_history(can_head, can_rel, can_tail, can_time, 'without_noise')
        pos_t_score_list, pos_ent_score_list, pos_rel_score_list = self.get_history(pos_head, pos_rel, pos_tail, pos_time, 'with_noise')
        if sample_type == 'r':
            can_score_r = np.sum(can_ent_score_list*can_rel_score_list)
            pos_score_r = np.sum(pos_ent_score_list*pos_rel_score_list)

            if str([pos_head, pos_rel, pos_tail, pos_time]) not in self.history_save_dict.keys():
                self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])] = []
            self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])].append(pos_score_r)

            predict_label = 0
            if can_score_r <= 0.1*np.min(self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])]):
                predict_label = -1
            if abs(can_score_r - pos_score_r) <= 0.01 or can_score_r - pos_score_r > 0:
                predict_label = 1
            
            return [can_head, can_rel, can_tail, can_time, predict_label, label, sample_type, can_score_r, str([pos_head, pos_rel, pos_tail, pos_time])]
        else:
            can_score_e = np.sum(can_ent_score_list*can_rel_score_list)
            pos_score_e = np.sum(pos_ent_score_list*pos_rel_score_list)

            if str([pos_head, pos_rel, pos_tail, pos_time]) not in self.history_save_dict.keys():
                self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])] = []
            self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])].append(pos_score_e)

            predict_label = 0
            if can_score_e <= 0.1*np.min(self.history_save_dict[str([pos_head, pos_rel, pos_tail, pos_time])]):
                predict_label = -1
            if abs(can_score_e - pos_score_e) <= 0.01 or can_score_e - pos_score_e > 0:
                predict_label = 1
            
            return [can_head, can_rel, can_tail, can_time, predict_label, label, sample_type, can_score_e, str([pos_head, pos_rel, pos_tail, pos_time])]

class TemporalDataset(object):
    def __init__(self, name: str):
        self.dataset = name
        self.root = Path(DATA_PATH) / name
        self.dtw_threshold = 10 # ICEWS14 20
        self.time_window = 10
        self.topk = 5

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates_no_reverse = int(maxis[1] + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2
        ## for yago11k and wikidata12k
        if self.data['valid'].shape[1]>4:
            self.interval = True # time intervals exist
            f = open(str(self.root / 'ts_id.pickle'), 'rb')
            self.time_dict = pickle.load(f)
        else:
            self.interval = False
            
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        try:
            inp_f = open(str(self.root / f'ts_diffs.pickle'), 'rb')
            self.time_diffs = torch.from_numpy(pickle.load(inp_f)).cuda().float()
            # print("Assume all timestamps are regularly spaced")
            # self.time_diffs = None
            inp_f.close()
        except OSError:
            print("Assume all timestamps are regularly spaced")
            self.time_diffs = None

        try:
            e = open(str(self.root / f'event_list_all.pickle'), 'rb')
            self.events = pickle.load(e)
            e.close()

            f = open(str(self.root / f'ts_id'), 'rb')
            dictionary = pickle.load(f)
            f.close()
            self.timestamps = sorted(dictionary.keys())
        except OSError:
            print("Not using time intervals and events eval")
            self.events = None

        if self.events is None:
            inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
            self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
            inp_f.close()

        self.e2te, self.e2r, self.e2e, self.s2r, self.o2r, self.r2r, self.e2fnum, self.p2num, self.f2num, self.r2to, self.r2ts, self.p2r, self.p2tr, self.e2tr, self.e2tre, self.r2s, self.r2o, self.e2r2e, self.t2e, self.t2p, self.triple2t = self.get_aux_dict('train')
        self.sot_check = set()

        # If dataset has events, it's wikidata.
        # For any relation that has no beginning & no end:
        # add special beginning = end = no_timestamp, increase n_timestamps by one.
        self.raw_weights = self.get_weight()

        self.check_set = set()
        self.train_check_set = set()
        self.valid_check_set = set()
        for file in [self.get_valid(), self.get_test()]:
            for sample in file:
                s = int(sample[0])
                r = int(sample[1])
                o = int(sample[2])
                t = int(sample[3])
                self.check_set.add(str([s,r,o,t]))
                
        for sample in self.get_train():
            s = int(sample[0])
            r = int(sample[1])
            o = int(sample[2])
            t = int(sample[3])
            self.train_check_set.add(str([s,r,o,t]))
        
        for sample in self.get_valid():
            s = int(sample[0])
            r = int(sample[1])
            o = int(sample[2])
            t = int(sample[3])
            self.valid_check_set.add(str([s,r,o,t]))

        self.filter_set = set()
    
    def get_aux_dict(self, file):
        train_set = self.get_train()
        e2t2e_dict = {}
        e2r2e_dict = {}
        e2t2r_dict = {}
        e2tre_dict = {}
        pair2r_dict = {}
        pair2t2r_dict = {}
        e2fnum = {}
        pair2fnum = {}
        f2tnum = {}
        sro2t = {}
        r2to = {}
        r2ts = {}
        e2r = {}
        e2e = {}
        s2r = {}
        o2r = {}
        r2r = {}
        r2s = {}
        r2o = {}
        t2e = {}
        t2p = {}

        max_num_e, max_num_p, max_num_f = 0, 0, 0
        for sample in tqdm(train_set):
            s = int(sample[0])
            r = int(sample[1])
            o = int(sample[2])
            t = int(sample[-1])

            # get t2e
            if t not in t2e.keys():
                t2e[t] = set()
            t2e[t].add(s)
            t2e[t].add(o)

            if str([s,r,o]) not in sro2t.keys():
                sro2t[str([s,r,o])] = []
            sro2t[str([s,r,o])].append(t)

            # get t2p
            if t not in t2p.keys():
                t2p[t] = set()
            t2p[t].add(str([s,o]))

            # get e2r2e dict
            if s not in e2r2e_dict.keys():
                e2r2e_dict[s] = {}
                e2r2e_dict[s][r] = set()
            else:
                if r not in e2r2e_dict[s].keys():
                    e2r2e_dict[s][r] = set()
            e2r2e_dict[s][r].add(o)

            # get r2s dict
            if r not in r2s.keys():
                r2s[r] = set()
            r2s[r].add(s)

            # get r2o dict
            if r not in r2o.keys():
                r2o[r] = set()
            r2o[r].add(o)

            # get e2t2r dict
            if s not in e2t2r_dict.keys():
                e2t2r_dict[s] = {}
                e2t2r_dict[s][t] = set()
            else:
                if t not in e2t2e_dict[s].keys():
                    e2t2r_dict[s][t] = set()
            e2t2r_dict[s][t].add(r)

            # get e2tre dict
            if s not in e2tre_dict.keys():
                e2tre_dict[s] = {}
                e2tre_dict[s][t] = set()
            else:
                if t not in e2tre_dict[s].keys():
                    e2tre_dict[s][t] = set()
            e2tre_dict[s][t].add(str([r, o]))

            # get r2to dict
            if r not in r2to.keys():
                r2to[r] = {}
                r2to[r][t] = set()
            else:
                if t not in r2to[r].keys():
                    r2to[r][t] = set()
            r2to[r][t].add(o)

            # get r2ts dict
            if r not in r2ts.keys():
                r2ts[r] = {}
                r2ts[r][t] = set()
            else:
                if t not in r2ts[r].keys():
                    r2ts[r][t] = set()
            r2ts[r][t].add(s)

            # get e2fnum dict
            if s not in e2fnum.keys():
                e2fnum[s] = 0
            if o not in e2fnum.keys():
                e2fnum[o] = 0
            e2fnum[s] += 1
            e2fnum[o] += 1

            if e2fnum[s] >= max_num_e:
                max_num_e = e2fnum[s]
            if e2fnum[o] >= max_num_e:
                max_num_e = e2fnum[o]
            
            # get pair2fnum dict
            if str([s,o]) not in pair2fnum.keys():
                pair2fnum[str([s,o])] = 0
            pair2fnum[str([s,o])] += 1

            if pair2fnum[str([s,o])] >= max_num_p:
                max_num_p = pair2fnum[str([s,o])]

            # get f2tnum dict
            if str([s,r,o]) not in f2tnum.keys():
                f2tnum[str([s,r,o])] = 0
            f2tnum[str([s,r,o])] += 1

            if f2tnum[str([s,r,o])] >= max_num_f:
                max_num_f = f2tnum[str([s,r,o])]

            # get e2r dict
            if s not in e2r.keys():
                e2r[s] = set()
            e2r[s].add(r)

            # get e2e dict
            if s not in e2e.keys():
                e2e[s] = set()
            e2e[s].add(o)

            if o not in e2e.keys():
                e2e[o] = set()
            e2e[o].add(s)

            # get s2r dict
            if s not in s2r.keys():
                s2r[s] = set()
            s2r[s].add(r)

            if o not in o2r.keys():
                o2r[o] = set()
            o2r[o].add(r)
                                    
            # get s2t2o dict
            if s not in e2t2e_dict.keys():
                e2t2e_dict[s] = {}
                e2t2e_dict[s][t] = set()
            else:
                if t not in e2t2e_dict[s].keys():
                    e2t2e_dict[s][t] = set()
            e2t2e_dict[s][t].add(o)

            # get pair2r dict
            if str([s,o]) not in pair2r_dict.keys():
                pair2r_dict[str([s,o])] = set()
            pair2r_dict[str([s,o])].add(r)

            # get pair2t2r dict
            if str([s,o]) not in pair2t2r_dict.keys():
                pair2t2r_dict[str([s,o])] = {}
                pair2t2r_dict[str([s,o])][t] = set()
            if t not in pair2t2r_dict[str([s,o])].keys():
                pair2t2r_dict[str([s,o])][t] = set()
            pair2t2r_dict[str([s,o])][t].add(r)

        # sorted e2t2e dict
        e2te = {}
        for s in e2t2e_dict.keys():
            raw_t2e = e2t2e_dict[s]
            e2te[s] = []
            for time in range(self.n_timestamps):
                if time in raw_t2e.keys():
                    e2te[s].append(raw_t2e[time])
                else:
                    e2te[s].append(set([]))

        # sorted e2t2r dict
        e2tr = {}
        for s in e2t2r_dict.keys():
            raw_t2r = e2t2r_dict[s]
            e2tr[s] = []
            for time in range(self.n_timestamps):
                if time in raw_t2r.keys():
                    e2tr[s].append(raw_t2r[time])
                else:
                    e2tr[s].append(set([]))
        
        # sorted r2to dict
        r2to_sort = {}
        for r in r2to.keys():
            raw_t2e = r2to[r]
            r2to_sort[r] = []
            for time in range(self.n_timestamps):
                if time in raw_t2e.keys():
                    r2to_sort[r].append(raw_t2e[time])
                else:
                    r2to_sort[r].append(set([]))
        
        # sorted r2ts dict
        r2ts_sort = {}
        for r in r2ts.keys():
            raw_t2e = r2ts[r]
            r2ts_sort[r] = []
            for time in range(self.n_timestamps):
                if time in raw_t2e.keys():
                    r2ts_sort[r].append(raw_t2e[time])
                else:
                    r2ts_sort[r].append(set([]))
        
        # sorted e2tre dict
        e2tre_sort = {}
        for s in e2tre_dict.keys():
            raw_t2re = e2tre_dict[s]
            e2tre_sort[s] = []
            for time in range(self.n_timestamps):
                if time in raw_t2re.keys():
                    e2tre_sort[s].append(raw_t2re[time])
                else:
                    e2tre_sort[s].append(set([]))
        
        # get r2r
        r2r = {}
        for pair_key in pair2t2r_dict.keys():
            t2r = pair2t2r_dict[pair_key]
            t_list = np.array(list(t2r.keys()))
            #print(t_list)
            t_index = np.argsort(t_list)
            #print(t_index)
            t_list = t_list[t_index]
            for index in range(len(t_list)):
                tmp_set = t2r[t_list[index]] #set()
                for r in tmp_set:
                    if r not in r2r.keys():
                        r2r[r] = tmp_set
        
        
        # get p2tr
        p2tr = {}
        for pair_key in pair2t2r_dict.keys():
            tr = pair2t2r_dict[pair_key]
            p2tr[pair_key] = []
            for t in range(self.n_timestamps):
                if t in tr.keys():
                    p2tr[pair_key].append(tr[t])
                else:
                    p2tr[pair_key].append(set([]))
        # get e2fnum_norm
        norm = np.log(max_num_e+1)
        e2fnum_norm = {}
        for key in e2fnum.keys():
            e2fnum_norm[key] = int(e2fnum[key]/norm)
        
        # get pair2fnum_norm
        norm = np.log(max_num_p+1)
        pair2fnum_norm = {}
        for key in pair2fnum.keys():
            pair2fnum_norm[key] = int(pair2fnum[key]/norm)
        
        # get f2tnum_norm
        norm = np.log(max_num_f+1)
        f2tnum_norm = {}
        for key in f2tnum.keys():
            f2tnum_norm[key] = int(f2tnum[key]/norm)

        return e2te, e2r, e2e, s2r, o2r, r2r, e2fnum_norm, pair2fnum_norm, f2tnum_norm, r2to_sort, r2ts_sort, pair2r_dict, p2tr, e2tr, e2tre_sort, r2s, r2o, e2r2e_dict, t2e, t2p, sro2t

    def get_examples(self, split):
        return self.data[split]

    def get_weight(self):
        print("generating raw weights for train samples...")
        train_set = self.data['train']
        train_weights = []
        
        for sample in tqdm(train_set):
            s = int(sample[0])
            r = int(sample[1])
            o = int(sample[2])
            t = int(sample[3])

            tmp_weight = np.array([0]*self.n_entities)
            FP_candidate_entities = [np.array(self.e2te[o][t])]
            FP_candidate_entities_flatten = set()
            for entity_set in FP_candidate_entities:
                FP_candidate_entities_flatten = FP_candidate_entities_flatten | entity_set
            tmp_weight[list(FP_candidate_entities_flatten)] = 1
            tmp_weight[o] = 0
            
            train_weights.append(tmp_weight)
        
        # for reverse sample
        for sample in tqdm(train_set):
            s = int(sample[0])
            r = int(sample[1])
            o = int(sample[2])
            t = int(sample[3])

            tmp_weight = np.array([0]*self.n_entities)
            FP_candidate_entities = [np.array(self.e2te[s][t])]
            FP_candidate_entities_flatten = set()
            for entity_set in FP_candidate_entities:
                FP_candidate_entities_flatten = FP_candidate_entities_flatten | entity_set
            tmp_weight[list(FP_candidate_entities_flatten)] = 1
            tmp_weight[s] = 0
            
            train_weights.append(tmp_weight)

        return torch.Tensor(train_weights)
    
    def get_test(self):
        copy = np.copy(self.data['test'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        return np.vstack((self.data['test'], copy))
    
    def get_valid(self):
        copy = np.copy(self.data['valid'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        return np.vstack((self.data['valid'], copy))
    
    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        return np.vstack((self.data['train'], copy))
    
    def select_tmp(self, tmp_dict, key, t, window):
        tmp_set_list = []
        source_list = tmp_dict[key]
        pre_index, aft_index = t-1, t+1
        num_count_pre, num_count_aft = 0, 0
        while pre_index >= 0 and num_count_pre < window:
            if len(source_list[pre_index]) == 0:
                pre_index = pre_index - 1
            else:
                tmp_set_list.append(source_list[pre_index])
                num_count_pre +=1
                pre_index = pre_index - 1
        
        while aft_index < self.n_timestamps and num_count_aft < window:
            if len(source_list[aft_index]) == 0:
                aft_index = aft_index + 1
            else:
                tmp_set_list.append(source_list[aft_index])
                num_count_aft +=1
                aft_index = aft_index + 1
        return tmp_set_list

    def filter_score(self, sample):
        s = int(sample[0])
        r = int(sample[1])
        o = int(sample[2])
        t = int(sample[3])

        if 'ICEWS' in self.dataset:
            lb , rb = 1, 10
        else:
            lb , rb = 0, 20

        r_check_num = 0
        s_check_num = 0
        o_check_num = 0
        t_check_num = 0
        all_r_num = 0
        all_s_num = 0
        all_o_num = 0
        all_t_num = 0

        valid_check_num_s = 0
        valid_check_num_o = 0
        valid_check_num_r = 0
        valid_check_num_t = 0

        # generate candidate samples by replacing entities and relations
        new_samples = []
        seen_r_list = self.r2r[r] #set()
        #seen_r_list = seen_r_list | self.s2r[s]
        #seen_r_list = seen_r_list | self.o2r[o]
        #seen_r_list = seen_r_list & self.r2r[r]

        seen_o_list = set()
        tmp_o = self.select_tmp(self.e2te, s, t, 1) #np.array(self.e2te[s])[max(0, t-self.time_window):min(self.n_timestamps-1, t+self.time_window)]
        for entity_set in tmp_o:
            seen_o_list = seen_o_list | entity_set

        seen_s_list = set()
        tmp_s = self.select_tmp(self.e2te, o, t, 1) #np.array(self.e2te[o])[max(0, t-self.time_window):min(self.n_timestamps-1, t+self.time_window)]
        for entity_set in tmp_s:
            seen_s_list = seen_s_list | entity_set

        if 'ICEWS' not in self.dataset:
            t_list = self.triple2t[str([s,r,o])]
            min_, max_ = min(t_list), max(t_list)
            tmp_t = []
            while min_ <= max_:
                tmp_t.append(min_)
                min_ += 1
        else:
            tmp_t = []
            if str([s, r, o, t+2]) in self.train_check_set:
                tmp_t.append(t+1)
            if str([s, r, o, t-2]) in self.train_check_set:
                tmp_t.append(t-1)
        tmp_new = []
        for t_ in tmp_t:
            if t_ < 0 or t_ >= self.n_timestamps:
                continue
            if abs(t_ - t) >= 1:
                continue 
            if str([s, r, o, t_]) in self.train_check_set:
                continue
            if str([s, r, o, t_]) in self.filter_set:
                continue
            if str([s, r, o, t_]) in self.check_set:
                tmp_new.append([s, r, o, t_, 1, 't'])
                self.filter_set.add(str([s, r, o, t_]))
            else:
                tmp_new.append([s, r, o, t_, 0, 't'])
                self.filter_set.add(str([s, r, o, t_]))
            if str([s, r, o, t_]) in self.valid_check_set:
                valid_check_num_t += 1
        for sample in tmp_new:
            new_samples.append(sample)
            all_t_num += 1
            if sample[-2] == 1:
                t_check_num += 1

        # replace r
        tmp_new = []
        for r_ in seen_r_list:
            if r_ not in self.e2r[s] or r_ not in self.e2r[o]:
                continue
            if str([s, r_, o, t]) in self.train_check_set:
                continue
            if str([s, r_, o, t]) in self.filter_set:
                continue
            if str([s, r_, o, t]) in self.check_set or str([s, r_, o, t-1]) in self.check_set or str([s, r_, o, t+1]) in self.check_set:
                tmp_new.append([s, r_, o, t, 1, 'r'])
                self.filter_set.add(str([s, r_, o, t]))
                self.filter_set.add(str([s, r_, o, t-1]))
                self.filter_set.add(str([s, r_, o, t+1]))
            else:
                tmp_new.append([s, r_, o, t, 0, 'r'])
                self.filter_set.add(str([s, r_, o, t]))
                self.filter_set.add(str([s, r_, o, t-1]))
                self.filter_set.add(str([s, r_, o, t+1]))
            if str([s, r_, o, t]) in self.valid_check_set:
                valid_check_num_r += 1
        for sample in tmp_new:
            new_samples.append(sample)
            all_r_num += 1
            if sample[-2] == 1:
                r_check_num += 1
        
        # replace s
        tmp_new = []
        for e_ in seen_s_list:
            if len(self.e2te[o][t]) > rb or len(self.e2te[o][t]) <= lb:
                continue 
            if r not in self.e2r[e_]:
                continue
            if str([e_, r, o, t]) in self.train_check_set:
                continue
            if str([e_, r, o, t]) in self.filter_set:
                continue
            if str([e_, r, o, t]) in self.check_set or str([e_, r, o, t-1]) in self.check_set or str([e_, r, o, t+1]) in self.check_set:
                tmp_new.append([e_, r, o, t, 1, 's'])
                self.filter_set.add(str([e_, r, o, t]))
                self.filter_set.add(str([e_, r, o, t-1]))
                self.filter_set.add(str([e_, r, o, t+1]))
            else:
                tmp_new.append([e_, r, o, t, 0, 's'])
                self.filter_set.add(str([e_, r, o, t]))
                self.filter_set.add(str([e_, r, o, t-1]))
                self.filter_set.add(str([e_, r, o, t+1]))
            if str([e_, r, o, t]) in self.valid_check_set:
                valid_check_num_s += 1
        for sample in tmp_new:
            new_samples.append(sample)
            all_s_num += 1
            if sample[-2] == 1:
                s_check_num += 1

        # replace o
        tmp_new = []
        for e_ in seen_o_list:
            if len(self.e2te[s][t]) > rb or len(self.e2te[s][t]) <= lb:
                continue 
            if r not in self.e2r[e_]:
                continue
            if str([s, r, e_, t]) in self.train_check_set:
                continue
            if str([s, r, e_, t]) in self.filter_set:
                continue
            if str([s, r, e_, t]) in self.check_set or str([s, r, e_, t-1]) in self.check_set or str([s, r, e_, t+1]) in self.check_set:
                tmp_new.append([s, r, e_, t, 1, 'o'])
                self.filter_set.add(str([s, r, e_, t]))
                self.filter_set.add(str([s, r, e_, t-1]))
                self.filter_set.add(str([s, r, e_, t+1]))
            else:
                tmp_new.append([s, r, e_, t, 0, 'o'])
                self.filter_set.add(str([s, r, e_, t]))
                self.filter_set.add(str([s, r, e_, t-1]))
                self.filter_set.add(str([s, r, e_, t+1]))
            if str([s, r, e_, t]) in self.valid_check_set:
                valid_check_num_o += 1
        for sample in tmp_new:
            new_samples.append(sample)
            all_o_num += 1
            if sample[-2] == 1:
                o_check_num += 1    
        
        return new_samples, s_check_num, r_check_num, o_check_num, t_check_num, all_s_num, all_r_num, all_o_num, all_t_num
    
    def get_booster_train_set(
            self, model: TKBCModel, pre_weights: torch.Tensor, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)):
        booster_predict_set = []
        s_check_all, r_check_all, o_check_all, t_check_all = 0, 0, 0, 0
        s_all, r_all, o_all, t_all = 0, 0, 0, 0
        booster_predict_set = []
        self.filter_set = set()
        train_data = self.get_train()
        for sample_index in tqdm(range(len(train_data))):
            pos_sample = train_data[sample_index]
            filtered_samples, s_check_num, r_check_num, o_check_num, t_check_num, all_s_num, all_r_num, all_o_num, all_t_num = self.filter_score(pos_sample)
            s_check_all += s_check_num
            r_check_all += r_check_num
            o_check_all += o_check_num
            t_check_all += t_check_num
            
            s_all += all_s_num
            r_all += all_r_num
            o_all += all_o_num
            t_all += all_t_num
            for tmp in filtered_samples:
                booster_predict_set.append([[int(pos_sample[0]), int(pos_sample[1]), int(pos_sample[2]), int(pos_sample[3])], tmp])
        '''
        print(booster_predict_set[0])
        print([s_check_all, r_check_all, o_check_all, t_check_all])
        print([s_all, r_all, o_all, t_all])
        print(sum([s_check_all, r_check_all, o_check_all, t_check_all]))
        print(sum([s_all, r_all, o_all, t_all]))
        print(float(sum([s_check_all, r_check_all, o_check_all, t_check_all]))/float(sum([s_all, r_all, o_all, t_all])))
        '''
        return None, booster_predict_set
    
    def eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), epoch_num: int=0):
        if self.events is not None:
            return self.time_eval(model, split, n_queries, 'rhs', at)
        test = self.get_examples(split)

        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        score_list = []
        mean_reciprocal_rank = {}
        hits_at = {}
        
        examples = torch.from_numpy(test.astype('int64')).cuda()
        for m in missing:
            q = examples.clone()
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks, scores = model.get_ranking(q, self.to_skip[m], batch_size=500)
            score_list.append(scores)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))
        '''
        if split == 'test':
            store_epoches = [30]
            if epoch_num in store_epoches:
                pickle.dump(ranks, open('/titan_data2/zhangjs/TELM_ranks/ta_rank_boo_e'+str(epoch_num)+'.pickle', 'wb'))
                print("saved epoch "+str(epoch_num))
        '''
        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps
