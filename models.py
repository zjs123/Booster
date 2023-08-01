from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from Utils import HingeLoss, BoosterLoss
import os
import tqdm
import math
import Utils
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking_score(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        rank_scores_list = []
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        q = self.get_queries(these_queries)
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    rank_scores_list.append(scores.cpu())
                   
                    b_begin += batch_size

                c_begin += chunk_size
        return torch.cat(rank_scores_list, 0)
    
    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        score_list = []
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)
                    scores = q @ rhs 
                    targets = self.score(these_queries)
                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):  
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    '''
                    for index in range(len(scores)):
                        #score_list.append([these_queries[index].cpu().detach().numpy(), scores[index].cpu().detach().numpy(), targets[index].cpu().detach().numpy(), ranks.detach().numpy()])
                        #score_list.append([these_queries[index].cpu().detach().numpy(),ranks.detach().numpy()])
                        score_list.append([ranks.detach().numpy()])
                    '''
                    b_begin += batch_size
                c_begin += chunk_size
        return ranks, score_list

class TeRo(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(TeRo, self).__init__()
        self.sizes = sizes
        self.rank = 500
        self.embedding_dim = rank
        self.n_entity = self.sizes[0]
        self.n_relation = self.sizes[1]
        self.n_time = self.sizes[3]
        self.L = 'L1'

        self.all_entities_list = [i for i in range(self.sizes[0])]
        self.all_entities = torch.LongTensor([i for i in range(self.sizes[0])]).cuda()

        self.emb_E_real = torch.nn.Embedding(self.n_entity, self.embedding_dim)
        self.emb_E_img = torch.nn.Embedding(self.n_entity, self.embedding_dim)
        self.emb_R_real = torch.nn.Embedding(self.n_relation*2, self.embedding_dim)
        self.emb_R_img = torch.nn.Embedding(self.n_relation*2, self.embedding_dim)
        self.emb_Time = torch.nn.Embedding(self.n_time, self.embedding_dim)
        
        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E_real.weight.data.uniform_(-r, r)
        self.emb_E_img.weight.data.uniform_(-r, r)
        self.emb_R_real.weight.data.uniform_(-r, r)
        self.emb_R_img.weight.data.uniform_(-r, r)
        self.emb_Time.weight.data.uniform_(-r, r)
        # self.emb_T_img.weight.data.uniform_(-r, r)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self, fact, skip_e):
        head, rel, tail, time = fact
        ret_facts = []
        test_e = set(self.all_entities_list) - set(skip_e) - set([tail])
        ret_facts = [[head, rel, i, time] for  i in test_e]
        ret_facts = [list(fact)] + ret_facts
        
        return torch.LongTensor(ret_facts).cuda()
    
    def score(self, x):
        h_i = x[0].repeat(self.sizes[0]+1)
        r_i = x[1].repeat(self.sizes[0]+1)
        t_i = torch.LongTensor([x[2]]+self.all_entities_list).cuda()
        d_i = x[3].repeat(self.sizes[0]+1)

        pi = 3.14159265358979323846
        d_img = torch.sin(self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(
            self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_img

        t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(t_i).view(-1,self.embedding_dim)*d_img


        r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

        h_img = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_img+\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_real


        t_img = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_img+\
                self.emb_E_img(t_i).view(-1,self.embedding_dim) *d_real

        r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)

        if self.L == 'L1':
            out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
            out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
            out = out_real + out_img

        else:
            out_real = torch.sum((h_real + r_real - t_real) ** 2, 1)
            out_img = torch.sum((h_img + r_img + t_real) ** 2, 1)
            out = torch.sqrt(out_img + out_real)

        return -out
	
    def forward(self, x, weight, type_):
        pi = 3.14159265358979323846
        if type_ == 'o': 
            h_i = x[:, 0].unsqueeze(-1).repeat(1, 50)
            r_i = x[:, 1].unsqueeze(-1).repeat(1, 50)
            t_i = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(h_i.size()[0])]).cuda()
            t_i = torch.cat([x[:, 2].unsqueeze(-1), t_i], -1)
            d_i = x[:, 3].unsqueeze(-1).repeat(1, 50)%3

        if type_ == 's':
            t_i = x[:, 2].unsqueeze(-1).repeat(1, 50)
            r_i = x[:, 1].unsqueeze(-1).repeat(1, 50)
            h_i = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(t_i.size()[0])]).cuda()
            h_i = torch.cat([x[:, 0].unsqueeze(-1), h_i], -1)
            d_i = x[:, 3].unsqueeze(-1).repeat(1, 50)%3

        d_img = torch.sin(self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(
            self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_real-\
                self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_img

        t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_real-\
                self.emb_E_img(t_i).view(-1,self.embedding_dim)*d_img


        r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

        h_img = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_img+\
                self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_real


        t_img = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_img+\
                self.emb_E_img(t_i).view(-1,self.embedding_dim) *d_real

        r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)

        if self.L == 'L1':
            out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
            out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
            out = out_real + out_img

        else:
            out_real = torch.sum((h_real + r_real - t_real) ** 2, 1)
            out_img = torch.sum((h_img + r_img + t_real) ** 2, 1)
            out = torch.sqrt(out_img + out_real)
        reg = (h_real, t_real, r_real, h_img, t_img, r_img, d_img, d_real)
        
        return out.view(-1, 50), reg, None
    
    def get_ranking(self, queries, filters, batch_size, year2id=None, chunk_size = -1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        score_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(queries))):
                fact = queries[i]
                can_scores = self.score(fact).cpu()
                filter_e = torch.LongTensor(filters[(int(fact[0]), int(fact[1]), int(fact[3]))])
                can_scores[filter_e] = 1e-6
                rank = self.getRank(can_scores)
                ranks[i] += rank
        return ranks, None    

class Timeplex_base(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(Timeplex_base, self).__init__()
        self.sizes = sizes
        self.rank = 200
        self.all_entities_list = [i for i in range(self.sizes[0])]
        self.all_entities = torch.LongTensor([i for i in range(self.sizes[0])]).cuda()

        init_embed = {}
        for embed_type in ["E_im", "E_re", "R_im", "R_re", "T_im", "T_re"]:
            init_embed[embed_type] = None
        self.entity_count = self.sizes[0]
        self.embedding_dim = rank
        self.relation_count = self.sizes[1]
        self.timeInterval_count = self.sizes[3]

        #self.has_cuda = True

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_im"] is None else \
            init_embed["E_im"]
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_re"] is None else \
            init_embed["E_re"]

        self.R_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_im"] is None else \
            init_embed["R_im"]
        self.R_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_re"] is None else \
            init_embed["R_re"]

        # E embeddingsfor (s,r,t) and (o,r,t) component
        self.E2_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.E2_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)

        # R embeddings for (s,r,t) component
        self.Rs_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Rs_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # R embeddings for (o,r,t) component
        self.Ro_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Ro_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # time embeddings for (s,r,t)
        self.Ts_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.Ts_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        # time embeddings for (o,r,t)
        self.To_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.To_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        ##
        self.pad_max = torch.tensor([self.timeInterval_count + 1]).cuda()
        self.pad_min = torch.tensor([0]).cuda()

        # '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.E2_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E2_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_im.weight.data, 0, 0.05)

        # init time embeddings
        torch.nn.init.normal_(self.Ts_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ts_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.To_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.To_im.weight.data, 0, 0.05)
        # '''

        self.minimum_value = -self.embedding_dim * self.embedding_dim

        self.unit_reg = False

        self.reg = 3
        print("Regularization value: in time_complex_fast: ", self.reg)

        # --srt, ort weights --#
        self.srt_wt = 5.0 
        self.ort_wt = 5.0 
        self.sot_wt = 5.0

        self.time_reg_wt = 1.0 # ICEWS14 1.0 ICEWS05 1.0
        self.emb_reg_wt = 0.005

        self.time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}
    
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self, fact, skip_e):
        head, rel, tail, time = fact
        ret_facts = []
        test_e = set(self.all_entities_list) - set(skip_e) - set([tail])
        ret_facts = [[head, rel, i, time] for  i in test_e]
        ret_facts = [list(fact)] + ret_facts
        
        return torch.LongTensor(ret_facts).cuda()
    
    def regularizer(self, s, r, o, t, reg_val=0):

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        ts_re = self.Ts_re(t)
        ts_im = self.Ts_im(t)
        to_re = self.To_re(t)
        to_im = self.To_im(t)

        ####
        s2_im = self.E2_im(s)
        s2_re = self.E2_re(s)
        o2_im = self.E2_im(o)
        o2_re = self.E2_re(o)

        rs_re = self.Rs_re(r)
        rs_im = self.Rs_im(r)
        ro_re = self.Ro_re(r)
        ro_im = self.Ro_im(r)

        ####

        # te_re = self.Te_re(t)
        # te_im = self.Te_im(t)
        if reg_val:
            self.reg = reg_val
        # print("CX reg", reg_val)

        #--time regularization--#
        time_reg = 0.0
        if self.time_reg_wt!=0:
            ts_re_all = (self.Ts_re.weight.unsqueeze(0))#[:, :-2, :])
            ts_im_all = (self.Ts_im.weight.unsqueeze(0))#[:, :-2, :])
            to_re_all = (self.To_re.weight.unsqueeze(0))#[:, :-2, :])
            to_im_all = (self.To_im.weight.unsqueeze(0))#[:, :-2, :])
            
            time_reg = self.time_regularizer(ts_re_all, ts_im_all) + self.time_regularizer(to_re_all, to_im_all) 
            time_reg *= self.time_reg_wt
        
        # ------------------#

        if self.reg == 2:
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2 + tr_re**2 + tr_im**2).sum()
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum() + (tr_re**2 + tr_im**2).sum()
            rs_sum = (rs_re ** 2 + rs_im ** 2).sum()
            ro_sum = (ro_re ** 2 + ro_im ** 2).sum()
            o2_sum = (o2_re ** 2 + o2_im ** 2).sum()
            s2_sum = (s2_re ** 2 + s2_im ** 2).sum()

            ts_sum = (ts_re ** 2 + ts_im ** 2).sum()
            to_sum = (to_re ** 2 + to_im ** 2).sum()


            ret = (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum() + ts_sum + to_sum + rs_sum + ro_sum
            ret = self.emb_reg_wt * (ret/ s.shape[0])


        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), 
                      torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2),
                      torch.sqrt(rs_re ** 2 + rs_im ** 2),
                      torch.sqrt(ro_re ** 2 + ro_im ** 2), 
                      torch.sqrt(ts_re ** 2 + ts_im ** 2),
                      torch.sqrt(to_re ** 2 + to_im ** 2)]
            factor_wt = [1, 1, 1, 1, 1, 1, 1]
            reg = 0
            for ele,wt in zip(factor,factor_wt):
                reg += wt* torch.sum(torch.abs(ele) ** 3)
            ret =  self.emb_reg_wt * (reg / s.shape[0])
        else:
            print("Unknown reg for complex model")
            assert (False)

        return ret + time_reg


    def normalize_complex(self, T_re, T_im):
        with torch.no_grad():
            re = T_re.weight
            im = T_im.weight
            norm = re ** 2 + im ** 2
            T_re.weight.div_(norm)
            T_im.weight.div_(norm)

        return

    def post_epoch(self):
        with torch.no_grad():
            self.Ts_re.weight.div_(torch.norm(self.Ts_re.weight, dim=-1, keepdim=True))
            self.Ts_im.weight.div_(torch.norm(self.Ts_im.weight, dim=-1, keepdim=True))
            self.To_re.weight.div_(torch.norm(self.To_re.weight, dim=-1, keepdim=True))
            self.To_im.weight.div_(torch.norm(self.To_im.weight, dim=-1, keepdim=True))

            
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
            
    
    def complex_3way_fullsoftmax(self, s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, embedding_dim):
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im)  # tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re * r_re - s_im * r_im)  # tmp2 = tmp2.view(-1,self.embedding_dim)
            
            if o is not None:  # o.shape[1] > 1:
                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)
            else:  # all ent as neg samples
                tmp1 = tmp1.view(-1, embedding_dim)
                tmp2 = tmp2.view(-1, embedding_dim)

                o_re_tmp = o_re.view(-1, embedding_dim).transpose(0, 1)
                o_im_tmp = o_im.view(-1, embedding_dim).transpose(0, 1)
                result = tmp1 @ o_im_tmp + tmp2 @ o_re_tmp
            # result.squeeze_()
        else:
            tmp1 = o_im * r_re - o_re * r_im;  # tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im * r_im + o_re * r_re;  # tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:  # s.shape[1] > 1:
                result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
            else:
                tmp1 = tmp1.view(-1, embedding_dim)
                tmp2 = tmp2.view(-1, embedding_dim)

                s_im_tmp = s_im.view(-1, embedding_dim).transpose(0, 1)
                s_re_tmp = s_re.view(-1, embedding_dim).transpose(0, 1)
                result = tmp1 @ s_im_tmp + tmp2 @ s_re_tmp
            # result.squeeze_()
        return result

    def complex_3way_simple(self, s_re, s_im, r_re, r_im, o_re, o_im):  # <s,r,o_conjugate> when dim(s)==dim(r)==dim(o)
        sro = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
        return sro.sum(dim=-1)
    
    def time_regularizer(self, t_re, t_im):
        t_re, t_im = t_re.squeeze(), t_im.squeeze()
        t_re_diff = t_re[1:] - t_re[:-1]
        t_im_diff = t_im[1:] - t_im[:-1]

        diff = torch.sqrt(t_re_diff**2 + t_im_diff**2 + 1e-9)**3
        return torch.sum(diff) / (t_re.shape[0] - 1)
    
    def score(self, x):
        s = x[0].repeat(self.sizes[0]+1)
        r = x[1].repeat(self.sizes[0]+1)
        o = torch.LongTensor([x[2]]+self.all_entities_list).cuda()
        t = x[3].repeat(self.sizes[0]+1)

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        # embeddings for s,r,t component
        rs_im = self.Rs_im(r)
        rs_re = self.Rs_re(r)

        # embeddings for o,r,t component
        ro_im = self.Ro_im(r)
        ro_re = self.Ro_re(r)

        '''
		##added extra 2 embeddings (padding) for semless time smoothing 
		Need to remove those extra embedding while calculating scores for all posibble time points
		##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
		as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
		'''
        t_re = self.Ts_re(t)
        t_im = self.Ts_im(t)
        #t2_re = self.To_re(t)
        #t2_im = self.To_im(t)

        sro = self.complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im)

        srt = self.complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)

        # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
        ort = self.complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

        # sot = complex_3way_simple(s_re, s_im,  t2_re, t2_im, o_re, o_im)
        sot = self.complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

        result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
        # result = srt

        return result
	
    def forward(self, x, weight, type_):
        if type_ == 's':
            s = x[:,0].unsqueeze(-1)
            r = x[:,1].unsqueeze(-1)
            o = None
            t = x[:,3].unsqueeze(-1)
            #t = t[:, :, self.time_index["t_s"]]
            
            #if (t.shape[-1] == len(self.time_index)):  # pick which dimension to index
            #    t = t[:, :, self.time_index["t_s"]]
            #else:
            #    t = t[:, self.time_index["t_s"], :]

            s_im = self.E_im(s)
            r_im = self.R_im(r)
            o_im = self.E_im.weight.unsqueeze(0)
            s_re = self.E_re(s)
            r_re = self.R_re(r)
            o_re = self.E_re.weight.unsqueeze(0)

            # embeddings for s,r,t component
            rs_im = self.Rs_im(r)
            rs_re = self.Rs_re(r)

            # embeddings for o,r,t component
            ro_im = self.Ro_im(r)
            ro_re = self.Ro_re(r)

            '''
            ##added extra 2 embeddings (padding) for semless time smoothing 
            Need to remove those extra embedding while calculating scores for all posibble time points
            ##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
            as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
            '''
            t_re = self.Ts_re(t)
            t_im = self.Ts_im(t)
            #t2_re = self.To_re(t)
            #t2_im = self.To_im(t)

            sro = self.complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)
                
            srt = self.complex_3way_fullsoftmax(s, r, t, s_re, s_im, rs_re, rs_im, t_re, t_im, self.embedding_dim)
            
            # ort = complex_3way_fullsoftmax(o, r, t, o_re, o_im, ro_re, ro_im, t_re, t_im, self.embedding_dim)
            ort = self.complex_3way_fullsoftmax(t, r, o, t_re, t_im, ro_re, ro_im, o_re, o_im, self.embedding_dim)

            # sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t2_re, t2_im, o_re, o_im,  self.embedding_dim)
            sot = self.complex_3way_fullsoftmax(s, t, o, s_re, s_im, t_re, t_im, o_re, o_im,  self.embedding_dim)

            # result = srt
            result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
            reg = self.regularizer(x[:,0],x[:,1],x[:,2],x[:,3])

            return result.view(-1, self.entity_count), reg
        
        if type_ == 'o':
            s = None
            r = x[:,1].unsqueeze(-1)
            o = x[:,2].unsqueeze(-1)
            t = x[:,3].unsqueeze(-1)
            #t = t[:, :, self.time_index["t_s"]]
            
            #if (t.shape[-1] == len(self.time_index)):  # pick which dimension to index
            #    t = t[:, :, self.time_index["t_s"]]
            #else:
            #    t = t[:, self.time_index["t_s"], :]

            s_im = self.E_im.weight.unsqueeze(0)
            r_im = self.R_im(r)
            o_im = self.E_im(o)
            s_re = self.E_re.weight.unsqueeze(0)
            r_re = self.R_re(r)
            o_re = self.E_re(o)

            # embeddings for s,r,t component
            rs_im = self.Rs_im(r)
            rs_re = self.Rs_re(r)

            # embeddings for o,r,t component
            ro_im = self.Ro_im(r)
            ro_re = self.Ro_re(r)

            '''
            ##added extra 2 embeddings (padding) for semless time smoothing 
            Need to remove those extra embedding while calculating scores for all posibble time points
            ##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
            as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
            '''
            t_re = self.Ts_re(t)
            t_im = self.Ts_im(t)
            #t2_re = self.To_re(t)
            #t2_im = self.To_im(t)

            sro = self.complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)
                
            srt = self.complex_3way_fullsoftmax(s, r, t, s_re, s_im, rs_re, rs_im, t_re, t_im, self.embedding_dim)
            
            # ort = complex_3way_fullsoftmax(o, r, t, o_re, o_im, ro_re, ro_im, t_re, t_im, self.embedding_dim)
            ort = self.complex_3way_fullsoftmax(t, r, o, t_re, t_im, ro_re, ro_im, o_re, o_im, self.embedding_dim)

            # sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t2_re, t2_im, o_re, o_im,  self.embedding_dim)
            sot = self.complex_3way_fullsoftmax(s, t, o, s_re, s_im, t_re, t_im, o_re, o_im,  self.embedding_dim)

            # result = srt
            result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
            reg = self.regularizer(x[:,0],x[:,1],x[:,2],x[:,3])

            return result.view(-1, self.entity_count), reg
    
    def get_ranking(self, queries, filters, batch_size, year2id=None, chunk_size = -1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        score_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(queries))):
                fact = queries[i]
                can_scores = self.score(fact).cpu()
                filter_e = torch.LongTensor(filters[(int(fact[0]), int(fact[1]), int(fact[3]))])
                can_scores[filter_e] = 1e-6
                rank = self.getRank(can_scores)
                ranks[i] += rank
        return ranks, None


class TA(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(TA, self).__init__()
        self.sizes = sizes
        self.rank = 200
        

        self.ent_embs = nn.Embedding(self.sizes[0], rank).cuda()
        self.rel_embs = nn.Embedding(self.sizes[1], rank).cuda()
        self.t_embs = nn.Embedding(self.sizes[3], rank).cuda()
        
        self.all_entities_list = [i for i in range(self.sizes[0])]
        self.all_entities = torch.LongTensor([i for i in range(self.sizes[0])]).cuda()
        
        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_linear = nn.Linear(4*rank, rank, bias = True)
        self.LSTM = nn.LSTM(input_size=rank, hidden_size=rank, num_layers=1)
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.t_embs.weight)

        normalize_entity_emb = F.normalize(self.ent_embs.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embs.weight.data, p=2, dim=1)
        normalize_temporal_emb = F.normalize(self.t_embs.weight.data, p=2, dim=1)
        self.ent_embs.weight.data = normalize_entity_emb
        self.rel_embs.weight.data = normalize_relation_emb
        self.t_embs.weight.data = normalize_temporal_emb
    
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self, fact, skip_e):
        head, rel, tail, time = fact
        ret_facts = []
        test_e = set(self.all_entities_list) - set(skip_e) - set([tail])
        ret_facts = [[head, rel, i, time] for  i in test_e]
        ret_facts = [list(fact)] + ret_facts
        
        return torch.LongTensor(ret_facts).cuda()
        
            
    def get_time_embedd(self, rels, time):

        #time_rel_emb = torch.tanh(self.time_linear(torch.cat([rels, y_emb, m_emb, d_emb], -1)))
        if len(time.size()) == 2:
            t_emb = self.t_embs(time)
            t_emb = t_emb.view(-1, self.rank).unsqueeze(0)
            rels = rels.view(-1, self.rank).unsqueeze(0)

            lstm_input = torch.cat([rels, t_emb], 0) # 4*batch_size_emb_dim
            output, _ = self.LSTM(lstm_input)
            time_rel_emb = output[0, :, :].view(-1, 50, self.rank)
            #time_rel_emb = (self.LSTM(lstm_input)[1][0][0]).view(-1, 50, self.rank)
        else:
            t_emb = self.t_embs(time).unsqueeze(0)
            rels = rels.unsqueeze(0)
            lstm_input = torch.cat([rels, t_emb], 0) # 4*batch_size_emb_dim
            output, _= self.LSTM(lstm_input)
            time_rel_emb = output[0, :, :]
            #time_rel_emb = self.LSTM(lstm_input)[1][0][0]

        return time_rel_emb

    def getEmbeddings(self, heads, rels, tails, time, intervals = None):
        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)        
        r_t = self.get_time_embedd(r, time)
        return h,r_t,t
    
    def score(self, x):

        s = x[0].repeat(self.sizes[0]+1)
        r = (x[1]).repeat(self.sizes[0]+1)
        o = torch.LongTensor([x[2]]+self.all_entities_list).cuda()
        time = x[3].repeat(self.sizes[0]+1)

        h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, time)
        scores = (h_embs * r_embs) * t_embs
        scores = torch.sum(scores, dim=-1)

        return scores
	
    def forward(self, x, weight, type_):
        if type_ == 'o': 
            s = x[:, 0].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)
            o = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(s.size()[0])]).cuda()
            o = torch.cat([x[:, 2].unsqueeze(-1), o], -1)
            time = x[:, 3].unsqueeze(-1).repeat(1, 50)

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, time)
            scores = (h_embs * r_embs) * t_embs
            scores = torch.sum(scores, dim=-1)
            reg_s = (r_embs)
            return scores, reg_s, None
        if type_ == 's':
            o = x[:, 2].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)
            s = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(o.size()[0])]).cuda()
            s = torch.cat([x[:, 0].unsqueeze(-1), s], -1)
            time = x[:, 3].unsqueeze(-1).repeat(1, 50)

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, time)
            scores = (h_embs * r_embs) * t_embs
            scores = torch.sum(scores, dim=-1)
            reg_o = (r_embs)
            return scores, reg_o, None
    
    def get_ranking(self, queries, filters, batch_size, year2id=None, chunk_size = -1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        score_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(queries))):
                fact = queries[i]
                can_scores = self.score(fact).cpu()
                filter_e = torch.LongTensor(filters[(int(fact[0]), int(fact[1]), int(fact[3]))])
                can_scores[filter_e] = 1e-6
                rank = self.getRank(can_scores)
                ranks[i] += rank
        return ranks, None

class HyTE(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(HyTE, self).__init__()
        self.sizes = sizes
        self.rank = 200

        self.ent_embs = nn.Embedding(self.sizes[0], rank).cuda()
        self.rel_embs = nn.Embedding(self.sizes[1], rank).cuda()
        self.time_embs =  nn.Embedding(self.sizes[3], rank).cuda()
        self.all_entities_list = [i for i in range(self.sizes[0])]
        self.all_entities = torch.LongTensor([i for i in range(self.sizes[0])]).cuda()
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.time_embs.weight)
    
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self, fact, skip_e):
        head, rel, tail, time = fact
        ret_facts = []
        test_e = set(self.all_entities_list) - set(skip_e) - set([tail])
        ret_facts = [[head, rel, i, time] for  i in test_e]
        ret_facts = [list(fact)] + ret_facts
        
        return torch.LongTensor(ret_facts).cuda()
        
            
    def get_time_embedd(self, entities, time):
        time_emb = entities - torch.sum(entities*time, -1, keepdim = True)*time
        return time_emb

    def getEmbeddings(self, heads, rels, tails, time, intervals = None):
        h,r,t,time = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails), self.time_embs(time)        
        h_t = self.get_time_embedd(h, time)
        t_t = self.get_time_embedd(t, time)
        r_t = self.get_time_embedd(r, time)
        
        return h_t,r_t,t_t
    
    def score(self, x):

        s = x[0].repeat(self.sizes[0]+1)
        r = (x[1]).repeat(self.sizes[0]+1)#%self.sizes[1]
        o = torch.LongTensor([x[2]]+self.all_entities_list).cuda()
        t = x[3].repeat(self.sizes[0]+1)

        h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, t)
        scores = torch.abs(h_embs + r_embs - t_embs)
        scores = torch.sum(scores, dim=-1)

        return scores
	
    def forward(self, x, weight, type_):
        if type_ == 'o': 
            s = x[:, 0].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)
            o = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(s.size()[0])]).cuda()
            o = torch.cat([x[:, 2].unsqueeze(-1), o], -1)
            t = x[:, 3].unsqueeze(-1).repeat(1, 50)

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, t)
            scores = torch.abs(h_embs + r_embs - t_embs)
            scores = F.dropout(scores, p= 0.4, training=self.training)
            scores = torch.sum(scores, dim=-1)
            reg_s = (self.ent_embs(x[:, 0]), self.rel_embs(x[:,1]), self.ent_embs(x[:, 2]))
            return scores, reg_s, self.time_embs(x[:,3])
        if type_ == 's':
            o = x[:, 2].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)
            s = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(o.size()[0])]).cuda()
            s = torch.cat([x[:, 0].unsqueeze(-1), s], -1)
            t = x[:, 3].unsqueeze(-1).repeat(1, 50)

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, t)
            scores = torch.abs(h_embs + r_embs - t_embs)
            scores = F.dropout(scores, p= 0.4, training=self.training)
            scores = torch.sum(scores, dim=-1)
            reg_o = (self.ent_embs(x[:, 0]), self.rel_embs(x[:,1]), self.ent_embs(x[:, 2]))
            return scores, reg_o, self.time_embs(x[:,3])
        
    def post_epoch(self):
        with torch.no_grad():
            self.time_embs.weight.div_(torch.norm(self.time_embs.weight, dim=-1, keepdim=True))
    
    def get_ranking(self, queries, filters, batch_size, year2id=None, chunk_size = -1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        score_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(queries))):
                fact = queries[i]
                can_scores = self.score(fact).cpu()
                filter_e = torch.LongTensor(filters[(int(fact[0]), int(fact[1]), int(fact[3]))])
                can_scores[filter_e] = 1e-20
                rank = self.getRank(can_scores)
                ranks[i] += rank
        return ranks, None

class DE(torch.nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(DE, self).__init__()
        self.sizes = sizes
        self.rank = 200
        

        self.ent_embs = nn.Embedding(self.sizes[0], int(rank*0.36)).cuda()
        self.rel_embs = nn.Embedding(self.sizes[1], rank).cuda()
        self.all_entities_list = [i for i in range(self.sizes[0])]
        self.all_entities = torch.LongTensor([i for i in range(self.sizes[0])]).cuda()
        
        # Creating and initializing the temporal embeddings for the entities 
        self.create_time_embedds()
        
        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
    
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self, facts):
        ret_facts = []
        for f in facts:
            s  = [f[0]]*len(self.all_entities_list)
            r  = [f[1]]*len(self.all_entities_list)
            o  = self.all_entities_list
            t  = [f[3]]*len(self.all_entities_list)
            ret_facts.append([s, r, o, t])
        
        return torch.LongTensor(ret_facts).cuda() # batch_size*4*N_entity

    def create_time_embedds(self):
        
        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.d_freq = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.y_freq = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.d_phi = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.y_phi = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        # amplitude embeddings for the entities
        self.m_amp = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.d_amp = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()
        self.y_amp = nn.Embedding(self.sizes[0], int(self.rank*0.64)).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        
            
    def get_time_embedd(self, entities, year, month, day):

        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))

        return y+m+d

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        if len(years.size()) == 1:
            years = years.view(-1,1)
            months = months.view(-1,1)
            days = days.view(-1,1)
        else:
            years = years.unsqueeze(-1)
            months = months.unsqueeze(-1)
            days = days.unsqueeze(-1)
        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)        
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)
        
        h = torch.cat((h,h_t), -1)
        t = torch.cat((t,t_t), -1)
        return h,r,t
    
    def score(self, x):

        s = x[0].repeat(self.sizes[0]+1)
        r = (x[1]).repeat(self.sizes[0]+1)%self.sizes[1]
        o = torch.LongTensor([x[2]]+self.all_entities_list).cuda()
        year = ((x[3] // 365) + 1).repeat(self.sizes[0]+1)
        month = (((x[3] % 365) // 30) + 1).repeat(self.sizes[0]+1)
        day = (((x[3] % 30)) + 1).repeat(self.sizes[0]+1)

        h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, year, month, day)
        scores = (h_embs * r_embs) * t_embs
        scores = torch.sum(scores, dim=-1)

        return scores
	
    def forward(self, x, weight, type_):
        if type_ == 'o': 
            s = x[:, 0].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)%self.sizes[1]
            o = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(s.size()[0])]).cuda()
            o = torch.cat([x[:, 2].unsqueeze(-1), o], -1)
            year = (x[:, 3].unsqueeze(-1).repeat(1, 50) // 365)+1
            month = (x[:, 3].unsqueeze(-1).repeat(1, 50) % 365) // 30+1
            day = (x[:, 3].unsqueeze(-1).repeat(1, 50) % 30)+1

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, year, month, day)
            scores = (h_embs * r_embs) * t_embs
            #scores = F.dropout(scores, p= 0.4, training=self.training)
            scores = torch.sum(scores, dim=-1)
            reg_s = (self.ent_embs(x[:,0]), self.y_freq(x[:, 0]), self.y_amp(x[:, 0]), self.y_phi(x[:, 0]), self.ent_embs(x[:,2]), self.y_freq(x[:, 2]), self.y_amp(x[:, 2]), self.y_phi(x[:, 2]), self.rel_embs(x[:,1]))
            return scores, reg_s, None
        if type_ == 's':
            o = x[:, 2].unsqueeze(-1).repeat(1, 50)
            r = x[:, 1].unsqueeze(-1).repeat(1, 50)%self.sizes[1]
            s = torch.LongTensor([random.choices(self.all_entities_list, k=49) for i in range(o.size()[0])]).cuda()
            s = torch.cat([x[:, 0].unsqueeze(-1), s], -1)
            year = (x[:, 3].unsqueeze(-1).repeat(1, 50) // 365)+1
            month = (x[:, 3].unsqueeze(-1).repeat(1, 50) % 365) // 30+1
            day = (x[:, 3].unsqueeze(-1).repeat(1, 50) % 30)+1

            h_embs, r_embs, t_embs = self.getEmbeddings(s, r, o, year, month, day)
            scores = (h_embs * r_embs) * t_embs
            #scores = F.dropout(scores, p= 0.4, training=self.training)
            scores = torch.sum(scores, dim=-1)
            reg_o = (self.ent_embs(x[:,0]), self.y_freq(x[:, 0]), self.y_amp(x[:, 0]), self.y_phi(x[:, 0]), self.ent_embs(x[:,2]), self.y_freq(x[:, 2]), self.y_amp(x[:, 2]), self.y_phi(x[:, 2]), self.rel_embs(x[:,1]))
            return scores, reg_o, None
    
    def get_ranking(self, queries, filters, batch_size, year2id=None, chunk_size = -1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        score_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(queries))):
                fact = queries[i]
                can_scores = self.score(fact).cpu()
                filter_e = torch.LongTensor(filters[(int(fact[0]), int(fact[1]), int(fact[3]))])
                can_scores[filter_e] = 1e-6
                rank = self.getRank(can_scores)
                ranks[i] += rank
        return ranks, None

class TNT(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(TNT, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.W = nn.Embedding(4*rank,1,sparse=False)
        self.W.weight.data *= 0

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=False)
            for s in [sizes[0], sizes[1], sizes[3]] # without no_time_emb
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        
        self.pre_train = pre_train
	
        if self.pre_train:
            self.embeddings[0].weight.data[:,self.rank:self.rank*3] *= 0
            self.embeddings[1].weight.data[:,self.rank:self.rank*3] *= 0
            self.embeddings[2].weight.data[:,self.rank:self.rank*3] *= 0
        

        self.no_time_emb = no_time_emb

        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True
	

    def score(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 
        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]
	
	    ## compute <h, r, T, t_conj> ==> 4**3
	    ## full_rel = r * time
        A =   rel[0]*time[0]+ rel[1]*time[1]+ rel[2]*time[2]- rel[3]*time[3] # scalar
        B =   rel[0]*time[1]+ rel[1]*time[0]- rel[2]*time[3]+ rel[3]*time[2] # e1
        C =   rel[0]*time[2]+ rel[2]*time[0]+ rel[1]*time[3]- rel[3]*time[1]  # e2
        D =   rel[1]*time[2]- rel[2]*time[1]+ rel[0]*time[3]+ rel[3]*time[0] # e1e2
	    
        full_rel = A,B,C,D
	    ## h * full_rel, note that we do not change +- sign here, thus we need do that later
        W =   lhs[0]*full_rel[0]+ lhs[1]*full_rel[1]+ lhs[2]*full_rel[2]- lhs[3]*full_rel[3] # scalar
        X =   lhs[0]*full_rel[1]+ lhs[1]*full_rel[0]- lhs[2]*full_rel[3]+ lhs[3]*full_rel[2] # e1
        Y =   lhs[0]*full_rel[2]+ lhs[2]*full_rel[0]+ lhs[1]*full_rel[3]- lhs[3]*full_rel[1]  # e2
        Z =   lhs[1]*full_rel[2]- lhs[2]*full_rel[1]+ lhs[0]*full_rel[3]+ lhs[3]*full_rel[0] # e1e2

        return torch.sum(W*rhs[0] - X * rhs[1] - Y * rhs[2] + Z * rhs[3], 1, keepdim=True)
	
	
	
    def forward(self, x):
        
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)

        lhs = lhs[:, :self.rank], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank*3:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank*3:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ to_score[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ to_score[1].t()
               ),(

                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else torch.cat((self.embeddings[2].weight[:,:self.rank],self.embeddings[2].weight[:,
	       self.rank*3:]),dim=1)


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity) 
        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]

        # compute <h, r, T, t_conj> ==> 4**4 / 2 
        # h * r
        A =   lhs[0]*rel[0]+ lhs[1]*rel[1]+ lhs[2]*rel[2]- lhs[3]*rel[3] # scalar
        B =   lhs[0]*rel[1]+ lhs[1]*rel[0]- lhs[2]*rel[3]+ lhs[3]*rel[2] # e1
        C =   lhs[0]*rel[2]+ lhs[2]*rel[0]+ lhs[1]*rel[3]- lhs[3]*rel[1]  # e2
        D =   lhs[1]*rel[2]- lhs[2]*rel[1]+ lhs[0]*rel[3]+ lhs[3]*rel[0] # e1e2
        # (h*r) * time, note that we first change the +- sign for easier dot product later
        W =   A * time[0]+ B * time[1]+ C * time[2]- D * time[3] # scalar
        X = - A * time[1]- B * time[0]+ C * time[3]- D * time[2] # e1
        Y = - A * time[2]- C * time[0]- B * time[3]+ D * time[1] # e2
        Z =   B * time[2]- C * time[1]+ A * time[3]+ D * time[0] # e1e2

        return torch.cat([W,X,Y,Z], 1)

    def get_lhs_queries(self, queries: torch.Tensor):
        rhs = self.embeddings[0](queries[:, 2])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity)
	
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]
	
        # compute <h, r, T, t_conj> ==> 4**3
        # full_rel = r * time
        A =   rel[0]*time[0]+ rel[1]*time[1]+ rel[2]*time[2]- rel[3]*time[3] # scalar
        B =   rel[0]*time[1]+ rel[1]*time[0]- rel[2]*time[3]+ rel[3]*time[2] # e1
        C =   rel[0]*time[2]+ rel[2]*time[0]+ rel[1]*time[3]- rel[3]*time[1]  # e2
        D =   rel[1]*time[2]- rel[2]*time[1]+ rel[0]*time[3]+ rel[3]*time[0] # e1e2

        full_rel = A,B,C,D
        
        # h * full_rel
	
        W1 =  full_rel[0]*rhs[0]- full_rel[1]*rhs[1]- full_rel[2]*rhs[2]+ full_rel[3]*rhs[3]
        X1 =  full_rel[1]*rhs[0]- full_rel[0]*rhs[1]- full_rel[3]*rhs[2]+ full_rel[2]*rhs[3]
        Y1 =  full_rel[2]*rhs[0]+ full_rel[3]*rhs[1]- full_rel[0]*rhs[2]- full_rel[1]*rhs[3]
        Z1 =- full_rel[3]*rhs[0]- full_rel[2]*rhs[1]+ full_rel[1]*rhs[2]+ full_rel[0]*rhs[3]
        return torch.cat([W1,X1,Y1,Z1], 1)
