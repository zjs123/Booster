import tqdm
import Utils
import torch
import pickle
import random
import numpy as np
from torch import nn
from torch import optim
from pathlib import Path

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = nn.CrossEntropyLoss(reduction='mean')
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions, factors, time = self.model.forward(input_batch)
                    truth = input_batch[:, 2]
                    l_fit = train_loss(predictions, truth)
                    l_reg = self.emb_regularizer.forward(factors)
                    l_time = torch.zeros_like(l_reg)
                    if time is not None:
                        l_time = self.temporal_regularizer.forward(time, self.model.W)
                            
                    #  fine-tune on critical samples
                    can_booster_predictions, can_factors, can_times = self.model.forward(can_sample_batch_booster)
                    can_truth = can_sample_batch_booster[:, 2]
                    l_fit_can = booster_loss(can_booster_predictions, can_truth, 'TNT')
                    l_reg_can = self.emb_regularizer.forward(can_factors)
                    l_time_can = torch.zeros_like(l_reg_can)
                    if can_times is not None:
                        l_time_can = self.temporal_regularizer.forward(can_times, self.model.W)

                    if 'ICEWS' in args.dataset:
                        l = l_fit + l_reg + l_time + l_fit_can + 0.1*l_reg_can + l_time_can
                    else:
                        l = l_fit + l_reg + l_time + l_fit_can# + 0.1*(l_reg_can + l_time_can)
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    bar.set_postfix(
                        loss=f'{l_fit.item():.0f}',
                        reg=f'{l_reg.item():.0f}',
                        cont=f'{l_time.item():.0f}'
                    )      
            
        else:
            train_loss = Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    
                    if pre_train:
                        predictions, factors, time = self.model.forward(input_batch)
                        truth = input_batch[:, 2]
                        l_fit = train_loss(predictions, truth, sample_weight, 'pre', 'TNT')
                        l_reg = self.emb_regularizer.forward(factors)
                        l_time = torch.zeros_like(l_reg)
                        if time is not None:
                            l_time = self.temporal_regularizer.forward(time, self.model.W)
                    else:
                        predictions, factors, time = self.model.forward(input_batch)
                        truth = input_batch[:, 2]
                        l_fit = train_loss(predictions, truth, sample_weight, 'fine-tune', 'TNT')
                        l_reg = self.emb_regularizer.forward(factors)
                        l_time = torch.zeros_like(l_reg)
                        if time is not None:
                            l_time = self.temporal_regularizer.forward(time, self.model.W)
                    
                    l = l_fit + l_reg + l_time
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    bar.set_postfix(
                        loss=f'{l_fit.item():.0f}',
                        reg=f'{l_reg.item():.0f}',
                        cont=f'{l_time.item():.0f}'
                    )


class DEOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = Utils.WeightCroLoss()
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                    predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                    truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                    l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'DE') + train_loss(predictions_o, truth, None, 'fine-tune', 'DE')
                    l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                            
                    #  fine-tune on critical samples
                    can_booster_predictions_o, can_factors_o, can_times_o = self.model.forward(can_sample_batch_booster, None, 'o')
                    can_booster_predictions_s, can_factors_s, can_times_s = self.model.forward(can_sample_batch_booster, None, 's')
                    can_truth = torch.zeros(can_booster_predictions_s.size()[0]).long().cuda()
                    l_fit_can = booster_loss(can_booster_predictions_s, can_truth, 'DE') + booster_loss(can_booster_predictions_o, can_truth, 'DE')
                    l_reg_can = self.emb_regularizer.forward(can_factors_s) + self.emb_regularizer.forward(can_factors_o)

                    l = l_fit + l_fit_can + 0.1*l_reg
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )        
            
        else:
            train_loss = Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ]
                    
                    if pre_train:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, sample_weight, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'pre', 'DE') + train_loss(predictions_o, truth, None, 'pre', 'DE')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)

                    else:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'DE') + train_loss(predictions_o, truth, None, 'fine-tune', 'DE')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                    
                    l = l_fit + 0.1*l_reg
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )


class HyTEOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = Utils.WeightCroLoss()
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                    predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                    truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                    if 'ICEWS' in args.dataset:
                        l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'HyTE') + train_loss(predictions_o, truth, None, 'fine-tune', 'HyTE')
                    else:
                        l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'HyTE') + train_loss(predictions_o, truth, None, 'fine-tune', 'HyTE')
                    l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                            
                    #  fine-tune on critical samples
                    can_booster_predictions_o, can_factors_o, can_times_o = self.model.forward(can_sample_batch_booster, None, 'o')
                    can_booster_predictions_s, can_factors_s, can_times_s = self.model.forward(can_sample_batch_booster, None, 's')
                    can_truth = torch.zeros(can_booster_predictions_s.size()[0]).long().cuda()
                    if 'ICEWS' in args.dataset:
                        l_fit_can = booster_loss(can_booster_predictions_s, can_truth, 'HyTE') + booster_loss(can_booster_predictions_o, can_truth, 'HyTE')
                    else:
                        l_fit_can = booster_loss(can_booster_predictions_s, can_truth, 'HyTE_ya') + booster_loss(can_booster_predictions_o, can_truth, 'HyTE_ya')
                    l_reg_can = self.emb_regularizer.forward(can_factors_s) + self.emb_regularizer.forward(can_factors_o)
                    l_time_can = torch.zeros_like(l_reg_can)

                    if 'ICEWS' in args.dataset:
                        l = l_fit + 0.1*l_reg + l_fit_can
                    else:
                        l = l_fit + l_fit_can# + 0.1*l_reg #+ l_reg_can
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    #self.model.post_epoch()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )        
            
        else:
            train_loss = Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ]
                    
                    if pre_train:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, sample_weight, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'pre', 'HyTE') + train_loss(predictions_o, truth, None, 'pre', 'HyTE')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                        l_time = torch.zeros_like(l_reg)

                    else:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        if 'ICEWS' in args.dataset:
                            l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'HyTE') + train_loss(predictions_o, truth, None, 'fine-tune', 'HyTE')
                        else:
                            l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'HyTE_ya') + train_loss(predictions_o, truth, None, 'fine-tune', 'HyTE_ya')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                        l_time = torch.zeros_like(l_reg)
                    
                    if 'ICEWS' in args.dataset:
                        l = l_fit + l_reg
                    else:
                        l = l_fit# + l_reg
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    #self.model.post_epoch()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )
             

class TAOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = nn.CrossEntropyLoss(reduction='mean')
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                    predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                    truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                    l_fit = train_loss(predictions_s, truth) + train_loss(predictions_o, truth)
                    l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                    l_time = torch.zeros_like(l_reg)
                            
                    #  fine-tune on critical samples
                    can_booster_predictions_o, can_factors_o, can_times_o = self.model.forward(can_sample_batch_booster, None, 'o')
                    can_booster_predictions_s, can_factors_s, can_times_s = self.model.forward(can_sample_batch_booster, None, 's')
                    can_truth = torch.zeros(can_booster_predictions_s.size()[0]).long().cuda()
                    l_fit_can = booster_loss(can_booster_predictions_s, can_truth, 'TA') + booster_loss(can_booster_predictions_o, can_truth, 'TA')
                    l_reg_can = self.emb_regularizer.forward(can_factors_s) + self.emb_regularizer.forward(can_factors_o)
                    l_time_can = torch.zeros_like(l_reg_can)

                    if 'ICEWS' in args.dataset:
                        l = l_fit + 0.1*l_fit_can
                    else:
                        l = l_fit + 0.1*l_fit_can
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )        
            
        else:
            train_loss = Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ]
                    
                    if pre_train:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, sample_weight, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'pre', 'TA') + train_loss(predictions_o, truth, None, 'pre', 'TA')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                        l_time = torch.zeros_like(l_reg)

                    else:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'TA') + train_loss(predictions_o, truth, None, 'fine-tune', 'TA')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                        l_time = torch.zeros_like(l_reg)
                    
                    
                    if 'ICEWS' in args.dataset:
                        l = l_fit
                    else:
                        l = l_fit + 0.001*l_reg
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )


class TimeplexOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = nn.CrossEntropyLoss(reduction='mean')
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions_s, reg_s = self.model.forward(input_batch, None, 's')
                    truth = input_batch[:, 2]
                    l_fit_s = train_loss(predictions_s, truth)#, sample_weight, 'fine-tune', 'Timeplex')

                    predictions_o, reg_o = self.model.forward(input_batch, None, 'o')
                    truth = input_batch[:, 0]
                    l_fit_o = train_loss(predictions_o, truth)#, None, 'fine-tune', 'Timeplex')

                    l_fit = l_fit_s+l_fit_o
                    l_reg = reg_s+reg_o
                            
                    #  fine-tune on critical samples
                    can_booster_predictions_s, can_reg_s = self.model.forward(can_sample_batch_booster, None, 's')
                    can_truth = can_sample_batch_booster[:, 2]
                    l_fit_can_s = booster_loss(can_booster_predictions_s, can_truth, 'Timeplex')

                    can_booster_predictions_o, can_reg_o = self.model.forward(can_sample_batch_booster, None, 'o')
                    can_truth = can_sample_batch_booster[:, 0]
                    l_fit_can_o = booster_loss(can_booster_predictions_o, can_truth, 'Timeplex')

                    l_fit_can = l_fit_can_s+l_fit_can_o
                    l_reg_can = can_reg_s+can_reg_o

                    l = l_fit + l_reg + l_fit_can + l_reg_can
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    self.model.post_epoch()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )        
            
        else:
            train_loss = nn.CrossEntropyLoss(reduction='mean') #Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    
                    if pre_train:
                        predictions_s, reg_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = input_batch[:, 2]
                        l_fit_s = train_loss(predictions_s, truth)#, None, 'pre', 'Timeplex')

                        predictions_o, reg_o = self.model.forward(input_batch, sample_weight, 'o')
                        truth = input_batch[:, 0]
                        l_fit_o = train_loss(predictions_o, truth)#, None, 'pre', 'Timeplex')

                        l_fit = l_fit_s+l_fit_o
                        l_reg = reg_s+reg_o

                    else:
                        predictions_s, reg_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = input_batch[:, 2]
                        l_fit_s = train_loss(predictions_s, truth)#, None, 'fine-tune', 'Timeplex')

                        predictions_o, reg_o = self.model.forward(input_batch, sample_weight, 'o')
                        truth = input_batch[:, 0]
                        l_fit_o = train_loss(predictions_o, truth)#, None, 'fine-tune', 'Timeplex')

                        l_fit = l_fit_s+l_fit_o
                        l_reg = reg_s+reg_o 

                    l = l_fit + l_reg
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    self.model.post_epoch()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )


class TeRoOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 1000,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.booster_pos_ratio = 0.5
        self.neg_ratio = 2

        self.root = Path('data/') / 'ICEWS14'
        in_file = open(str(self.root / ('test' + '.pickle')), 'rb')
        self.test = pickle.load(in_file)
        in_file = open(str(self.root / ('valid' + '.pickle')), 'rb')
        self.valid = pickle.load(in_file)

        self.test_batch_s_r_t = []
        self.test_batch = []
        self.test_weight = []
        for i in range(len(self.test)):
            s = self.test[i][0]
            r = self.test[i][1]
            o = self.test[i][2]
            t = self.test[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        for i in range(len(self.valid)):
            s = self.valid[i][0]
            r = self.valid[i][1]
            o = self.valid[i][2]
            t = self.valid[i][3]

            self.test_batch.append(str([s,r,o,t]))
            self.test_batch.append(str([o,r+230,s,t]))
            self.test_batch_s_r_t.append([s,t])
            self.test_weight.append(1)
        
        self.test_e2e = {}
        test_set = self.test
        valid_set = self.valid
        for sample in test_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        for sample in valid_set:
            h_e = sample[0]
            r = sample[1]
            t_e = sample[2]
            if h_e not in self.test_e2e.keys():
                self.test_e2e[h_e] = []
            self.test_e2e[h_e].append(t_e)
            if t_e not in self.test_e2e.keys():
                self.test_e2e[t_e] = []
            self.test_e2e[t_e].append(h_e)
        
        self.check = []
    
    def generate_booster_batch(self, booster_samples, booster_batch_size):
        sample_key = random.sample(booster_samples.keys(), booster_batch_size)
        anchor = []
        can = []
        neg = []

        for key in sample_key:
            can_list = np.array(booster_samples[key]['can'])
            neg_list = np.array(booster_samples[key]['neg'])
            can_index = np.random.choice(range(len(can_list)), 20)
            neg_index = np.random.choice(range(len(neg_list)), 20)
            can_sample = can_list[can_index]
            neg_sample = neg_list[neg_index]
            anchor.append(eval(key))
            can.append(can_sample)
            neg.append(neg_sample)

        return torch.LongTensor(anchor).cuda(), torch.LongTensor(can).view(-1, len(anchor[0])).cuda(), torch.LongTensor(neg).view(-1, len(anchor[0])).cuda()
        
    def epoch(self, examples, weights, booster_sample, args, pre_train):
        if len(booster_sample) != 0:
            booster_batch_size = int(len(booster_sample)/(examples.shape[0]//self.batch_size+1))
            train_loss = Utils.WeightCroLoss()
            booster_loss = Utils.BoosterLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss fine-turn')
                b_begin = 0
                b_begin_booster = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    can_sample_batch_booster = booster_sample[
                        b_begin_booster:b_begin_booster+booster_batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ].cuda()

                    # still training on easy samples
                    predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                    predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                    truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                    l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'TeRo') + train_loss(predictions_o, truth, None, 'fine-tune', 'TeRo')
                    l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                            
                    #  fine-tune on critical samples
                    can_booster_predictions_o, can_factors_o, can_times_o = self.model.forward(can_sample_batch_booster, None, 'o')
                    can_booster_predictions_s, can_factors_s, can_times_s = self.model.forward(can_sample_batch_booster, None, 's')
                    can_truth = torch.zeros(can_booster_predictions_s.size()[0]).long().cuda()
                    l_fit_can = booster_loss(can_booster_predictions_s, can_truth, 'TeRo') + booster_loss(can_booster_predictions_o, can_truth, 'TeRo')
                    l_reg_can = self.emb_regularizer.forward(can_factors_s) + self.emb_regularizer.forward(can_factors_o)

                    l = l_fit# + 0.1*l_reg + 0.1*l_fit_can #+ 0.1*(l_reg+l_reg_can)
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    b_begin_booster += booster_batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )        
            
        else:
            train_loss = Utils.WeightCroLoss()
            rand_index = torch.randperm(examples.shape[0])
            actual_examples = examples[rand_index, :]
            weights = weights[rand_index, :]
            with tqdm.tqdm(total=actual_examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss pre-train')
                b_begin = 0
                while b_begin < actual_examples.shape[0]:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                    sample_weight = weights[
                        b_begin:b_begin + self.batch_size
                    ]
                    
                    if pre_train:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, sample_weight, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, sample_weight, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'pre', 'TeRo') + train_loss(predictions_o, truth, None, 'pre', 'TeRo')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)

                    else:
                        predictions_o, factors_o, time_o = self.model.forward(input_batch, None, 'o')
                        predictions_s, factors_s, time_s = self.model.forward(input_batch, None, 's')
                        truth = torch.zeros(predictions_s.size()[0]).long().cuda()
                        l_fit = train_loss(predictions_s, truth, None, 'fine-tune', 'TeRo') + train_loss(predictions_o, truth, None, 'fine-tune', 'TeRo')
                        l_reg = self.emb_regularizer.forward(factors_s) + self.emb_regularizer.forward(factors_o)
                    
                    l = l_fit# + 0.1*l_reg
                    
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                    try:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{l_reg.item():.0f}',
                            cont=f'{l_time.item():.0f}'
                        )
                    except:
                        bar.set_postfix(
                            loss=f'{l_fit.item():.0f}',
                            reg=f'{0:.0f}',
                            cont=f'{0:.0f}'
                        )

