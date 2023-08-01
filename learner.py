import heapq
import argparse
from typing import Dict
import logging
import torch
import pickle
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from datasets import TemporalDataset, DisDataset
from optimizers import TKBCOptimizer, DEOptimizer, HyTEOptimizer, TAOptimizer, TimeplexOptimizer, TeRoOptimizer
from models import TNT, DE, HyTE, TA, Timeplex_base, TeRo
from regularizers import N3, N2, N1, Linear3
from tqdm import tqdm
import numpy as np
import os


parser = argparse.ArgumentParser(
    description="Booster"
)
parser.add_argument(
    '--dataset', default='ICEWS14', type=str,
    help="Dataset name"
)

parser.add_argument(
    '--model', default='TNT', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=1000, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=20, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--rank', default=200, type=int,
    help="Factorization rank." # 2000 for ICEWS14_TNT 200 for others
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size." # 1000 for ICEWS14_TNT 100 for ICEWS14_DE
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--time_granularity', default=1, type=int, 
    help="Time granularity for time embeddings"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def get_PRF(can_scores, pos_scores, labels, samples):
    can_scores = can_scores.cpu().detach().numpy() # batch_size*1
    pos_scores = pos_scores.cpu().detach().numpy() # batch_size*1
    labels = labels.cpu().detach().numpy() # batch_size
    samples = [element.cpu().detach().numpy() for element in samples] # [ batch_size*1, _, _, _, _, _]
    booster_samples = [] # [[s,r,o,t]]
    booster_score = []
    booster_label = []
    for index in range(len(can_scores)):
        # generate predict labels
        if can_scores[index] > max(pos_scores[index]):
        #if labels[index] == 1:
            booster_samples.append([samples[0][index], samples[1][index], samples[2][index], samples[3][index]])
            booster_score.append(can_scores[index])
            booster_label.append(labels[index])

    return booster_samples, booster_score, booster_label

def filter_booster_sample(score, sample, label):
    #return 0,0,0, sample
    pos_catch, neg_catch, all_pos = 0, 0, 0
    filter_sample = []
    topk_index = heapq.nlargest(15000, range(len(score)), score.__getitem__)
    sample = np.array(sample)[topk_index]
    for i in tqdm(range(len(sample))):
        filter_sample.append(sample[i])
        if label[i] == 1:
            pos_catch += 1
        else:
            neg_catch += 1
        if label[i] == 1:
            all_pos += 1
    return pos_catch, neg_catch, all_pos, filter_sample

def second_step_filter(pos_2_score_sample_dict):

    for pos_key in tqdm(pos_2_score_sample_dict):
        tmp_fact = eval(pos_key)
        can_list = pos_2_score_sample_dict[pos_key]['can']
        neg_list = pos_2_score_sample_dict[pos_key]['neg']
        if len(can_list) == 0:
            pos_2_score_sample_dict[pos_key]['can'].append(tmp_fact)
        if len(neg_list) == 0:
            tmp_neg_fact = [tmp_fact[0], tmp_fact[1], tmp_fact[0], tmp_fact[3]]
            pos_2_score_sample_dict[pos_key]['neg'].append(tmp_neg_fact)
    return pos_2_score_sample_dict

def booster_process(predict_data, dataset_name):
    print('caculating structured scores...')

    train_dataset = DisDataset(predict_data, dataset_name, Type='predict')
    train_dataloader = DataLoader(train_dataset, batch_size=1000, num_workers=40, shuffle = False, pin_memory = True)

    booster_sample_dict = {}
    all_pos = []
    all_s, all_r, all_o, all_t = 0, 0, 0, 0 # candidate sample number of each type
    pos_s, pos_r, pos_o, pos_t = 0, 0, 0, 0 # pos sample number of each type
    catch_s, catch_r, catch_o, catch_t = 0, 0, 0, 0 # selected as pos sample number of each type
    pos_catch_s, pos_catch_r, pos_catch_o, pos_catch_t = 0, 0, 0, 0  # selected as pos sample and actually pos number of each type
    for batch in tqdm(train_dataloader):
        for index in range(len(batch[0])):
            can_s = int(batch[0][index])
            can_r = int(batch[1][index])
            can_o = int(batch[2][index])
            can_t = int(batch[3][index])
            y_label = int(batch[4][index])
            true_label = int(batch[5][index])
            sample_type = batch[6][index]
            score = float(batch[7][index])
            pos_str = batch[8][index]

            if pos_str not in booster_sample_dict.keys():
                booster_sample_dict[pos_str] = {}
                booster_sample_dict[pos_str]['can'] = []
                booster_sample_dict[pos_str]['neg'] = []
            if y_label == 1:
                all_pos.append([can_s,can_r,can_o,can_t])
                booster_sample_dict[pos_str]['can'].append([can_s,can_r,can_o,can_t])
            if y_label == -1:
                booster_sample_dict[pos_str]['neg'].append([can_s,can_r,can_o,can_t])

            if sample_type == 's':
                all_s += 1
                if true_label == 1:
                    pos_s += 1
                if y_label == 1:
                    catch_s += 1
                if true_label == 1 and y_label == 1:
                    pos_catch_s += 1
            if sample_type == 'r':
                all_r += 1
                if true_label == 1:
                    pos_r += 1
                if y_label == 1:
                    catch_r += 1
                if true_label == 1 and y_label == 1:
                    pos_catch_r += 1
            if sample_type == 'o':
                all_o += 1
                if true_label == 1:
                    pos_o += 1
                if y_label == 1:
                    catch_o += 1
                if true_label == 1 and y_label == 1:
                    pos_catch_o += 1
            if sample_type == 't':
                all_t += 1
                if true_label == 1:
                    pos_t += 1
                if y_label == 1:
                    catch_t += 1
                if true_label == 1 and y_label == 1:
                    pos_catch_t += 1

    booster_sample_dict = second_step_filter(booster_sample_dict)
    '''
    print([all_s, all_r, all_o, all_t])
    print([pos_s, pos_r, pos_o, pos_t])
    print([catch_s, catch_r, catch_o, catch_t])
    print([pos_catch_s, pos_catch_r, pos_catch_o, pos_catch_t])
    print('recall: ' + str(float(sum([pos_catch_s, pos_catch_r, pos_catch_o, pos_catch_t])) / float(sum([pos_s, pos_r, pos_o, pos_t])+1)))
    print('precision: ' + str(float(sum([pos_catch_s, pos_catch_r, pos_catch_o, pos_catch_t])) / float(sum([catch_s, catch_r, catch_o, catch_t]))))
    '''
    return torch.LongTensor(all_pos)

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          time_granularity=args.time_granularity,
         ):


    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset

    ##restore model parameters and results
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/time_granularity{:02d}/emb_reg{:.5f}/time_reg{:.5f}/'.format(rank,learning_rate,batch_size, time_granularity, emb_reg, time_reg))
    
    dataset = TemporalDataset(dataset)
    
    sizes = dataset.get_shape()
    model = {
        'TNT': TNT(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'DE': DE(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'HyTE': HyTE(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'TA': TA(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'Timeplex': Timeplex_base(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'TeRo': TeRo(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity)
    }[model]
    model = model.cuda()


    opt = optim.Adagrad(model.parameters(), lr=0.5*learning_rate)
    opt_pretrain = optim.Adagrad(model.parameters(), lr=learning_rate)
    
    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg, "time_granularity =", time_granularity)
  
    # Results related
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    #os.makedirs(PATH)
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    epoch_pretrain = 10
    pre_weights = dataset.raw_weights
    examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )
    if args.model == 'TNT':
        emb_reg = N3(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = TKBCOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=batch_size
            )
        optimizer_pretrain = TKBCOptimizer(
            model, emb_reg, time_reg, opt_pretrain,
            batch_size=batch_size
        )
    if args.model == 'DE':
        emb_reg = N3(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = DEOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=batch_size
            )
        optimizer_pretrain = DEOptimizer(
            model, emb_reg, time_reg, opt_pretrain,
            batch_size=batch_size
        )
    if args.model == 'HyTE':
        emb_reg = N2(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = HyTEOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=batch_size
            )
        optimizer_pretrain = HyTEOptimizer(
            model, emb_reg, time_reg, opt_pretrain,
            batch_size=batch_size
        )
    if args.model == 'TA':
        emb_reg = N3(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = TAOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=batch_size
            )
        optimizer_pretrain = TAOptimizer(
            model, emb_reg, time_reg, opt_pretrain,
            batch_size=batch_size
        )
    if args.model == 'Timeplex':
        emb_reg = N2(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = TimeplexOptimizer(
                model, emb_reg, time_reg, optim.Adagrad(model.parameters(), lr=learning_rate),
                batch_size=batch_size
            )
        optimizer_pretrain = TimeplexOptimizer(
        model, emb_reg, time_reg, optim.Adagrad(model.parameters(), lr=learning_rate),
        batch_size=batch_size)
        #optimizer = optimizer_pretrain
    if args.model == 'TeRo':
        emb_reg = N2(emb_reg)
        time_reg = Linear3(time_reg)
        optimizer = TeRoOptimizer(
                model, emb_reg, time_reg, optim.Adagrad(model.parameters(), lr=learning_rate),
                batch_size=batch_size
            )
        optimizer_pretrain = TeRoOptimizer(
        model, emb_reg, time_reg, optim.Adagrad(model.parameters(), lr=learning_rate),
        batch_size=batch_size)
        optimizer = optimizer_pretrain
    booster_sample = []
    best_result = None
    mrr_best = 0
    for epoch in range(args.max_epochs):
        model.train()
        print("[ Epoch:", epoch, "]")

        if epoch >= epoch_pretrain:
            optimizer.epoch(examples,pre_weights,booster_sample,args,pre_train=False)
        else:
            optimizer_pretrain.epoch(examples,pre_weights,booster_sample,args,pre_train=True)
        
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            print('eval.......')
            valid, test = [
                # avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, use_left_queries=args.use_left))
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, epoch_num = epoch+1))
                for split in ['valid', 'test']
            ]
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            #print("train: ", train['MRR'])

            # Save results
            f = open(os.path.join(PATH, 'result.txt'), 'w+')
            f.write("\n VALID: ")
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']
            mrr_test = test['MRR']
            if mrr_test > mrr_best:
                mrr_best = mrr_test
                best_result = test

            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10 and epoch>epoch_pretrain:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            #if not dataset.interval:
            #    curve['train'].append(train)
            print("\t VALID : ", valid)
            print("\t TEST : ", test)
            
        if epoch == 20:
            print('get training set for booster ...')
            try:
                booster_predict_set = pickle.load(open("predict_set_"+args.dataset+".pickle", "rb"))
                print("find saved file")
            except:
                pre_weights_new, booster_predict_set = dataset.get_booster_train_set(model, pre_weights, 'train', -1)
                pickle.dump(booster_predict_set, open("predict_set_"+args.dataset+".pickle", "wb"))
            booster_sample = booster_process(booster_predict_set, datasetname)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    #print("\n\nTEST : ", results)
    print("\n\nTEST : ", best_result)
    f = open(os.path.join(PATH, 'result.txt'), 'w+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':

    learn()


