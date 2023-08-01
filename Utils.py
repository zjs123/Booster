# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:04:41 2020

@author: zjs
"""
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from fastdtw import fastdtw

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

class WeightCroLoss(nn.Module):
	def __init__(self):
		super(WeightCroLoss,self).__init__()
		bce_loss = nn.BCELoss()
	
	def masked_softmax(self, A, dim):
		A = A.float()
		A_max = torch.max(A,dim=dim,keepdim=True)[0]
		A_exp = torch.exp(A-A_max)
		A_exp = A_exp * (A != 0).float()
		Sum = torch.sum(A_exp,dim=dim,keepdim=True)
		Sum = Sum + (Sum == 0.0).float()
		score = A_exp / Sum

		#score = nn.Softmax(dim)(A)

		return score
    

	def forward(self, outputs, traget, sample_weight, Type, model_type):
		if sample_weight != None:
			if Type == 'pre':
				outputs_masked = -1e9*sample_weight+outputs
				log = F.log_softmax(outputs_masked, dim=1)

				traget = traget.reshape(-1,1)
				log = log.gather(1, traget)
				result_loss = -1*log
				result_loss = result_loss.mean(dim=0)
			else:
				outputs_masked = -1e9*sample_weight+outputs
				log = F.log_softmax(outputs_masked, dim=1)
				
				traget = traget.reshape(-1,1)
				log = log.gather(1, traget)
				result_loss = -1*log
				result_loss = result_loss.mean(dim=0)

				tar_outputs = outputs_masked.gather(1, traget)
				fine_tune_score_neg = (self.masked_softmax(torch.relu(-tar_outputs+outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked - tar_outputs.detach() - 2)).sum(dim = 1)
				result_loss += -fine_tune_score_neg.mean()
				
		else:
			if Type == 'pre':
				if model_type == 'TeRo':
					y_pos = outputs[:, 0]
					y_neg = outputs[:, 1:]
					M = y_pos.size(0)
					y_pos = 110-y_pos
					y_neg = 110-y_neg
					y_neg = y_neg.view(49, -1).transpose(0, 1)
					p = F.softmax(0.5*y_neg)
					loss_pos = torch.sum(F.softplus(-1 * y_pos))
					loss_neg = torch.sum(p * F.softplus(y_neg))
					result_loss = (loss_pos + loss_neg) / 2 / M
				else:
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)

					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)
			else:
				if model_type == 'TNT':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)

					tar_outputs = outputs_masked.gather(1, traget)
					fine_tune_score_neg = (self.masked_softmax(torch.relu(-tar_outputs+outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked - tar_outputs.detach() - 2)).sum(dim = 1)
					result_loss += -fine_tune_score_neg.mean()
				if model_type == 'HyTE':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)

					tar_outputs = outputs_masked.gather(1, traget)
					max_ = torch.max(outputs_masked,dim=1)[0]
					fine_tune_score_neg = (self.masked_softmax(torch.relu(tar_outputs-outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked/5)).sum(dim = 1)
					#result_loss += -fine_tune_score_neg.mean()

					fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-tar_outputs + max_), dim=0).detach())*F.logsigmoid((tar_outputs - max_.detach())/10)
					result_loss += -0.1*fine_tune_score_pos.mean()
				if model_type == 'HyTE_ya':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)

					tar_outputs = outputs_masked.gather(1, traget)
					max_ = torch.max(outputs_masked,dim=1)[0]
					fine_tune_score_neg = (self.masked_softmax(torch.relu(-tar_outputs+outputs_masked), dim=1).detach()*F.logsigmoid(-outputs_masked/5)).sum(dim = 1)
					result_loss += -fine_tune_score_neg.mean()

					fine_tune_score_pos = (self.masked_softmax(torch.relu(tar_outputs - max_), dim=0).detach())*F.logsigmoid((tar_outputs))
					#result_loss += -fine_tune_score_pos.mean()
				
				if model_type == 'DE':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)

					tar_outputs = outputs_masked.gather(1, traget)
					max_ = torch.max(outputs_masked,dim=1)[0]
					fine_tune_score_neg = (self.masked_softmax(torch.relu(tar_outputs-outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked/5)).sum(dim = 1)
					result_loss += -fine_tune_score_neg.mean()
					
					fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-tar_outputs + max_), dim=0).detach())*F.logsigmoid(tar_outputs*10 - max_.detach())
					result_loss += -fine_tune_score_pos.mean()
				if model_type == 'TA':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)
					
					tar_outputs = outputs_masked.gather(1, traget)
					max_ = torch.max(outputs_masked,dim=1)[0]
					fine_tune_score_neg = (self.masked_softmax(torch.relu(-tar_outputs+outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked/5)).sum(dim = 1)
					#result_loss += -fine_tune_score_neg.mean()
					
					fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-tar_outputs + max_), dim=0).detach())*F.logsigmoid(tar_outputs*10 - max_.detach())
					result_loss += -fine_tune_score_pos.mean()
				if model_type == 'Timeplex':
					outputs_masked = outputs
					log = F.log_softmax(outputs_masked, dim=1)
					
					traget = traget.reshape(-1,1)
					log = log.gather(1, traget)
					result_loss = -1*log
					result_loss = result_loss.mean(dim=0)

					tar_outputs = outputs_masked.gather(1, traget)
					max_ = torch.max(outputs_masked,dim=1)[0]
					fine_tune_score_neg = (self.masked_softmax(torch.relu(tar_outputs-outputs_masked+1), dim=1).detach()*F.logsigmoid(-outputs_masked - tar_outputs.detach() - 2)).sum(dim = 1)
					result_loss += -fine_tune_score_neg.mean()

					fine_tune_score_pos = (self.masked_softmax(torch.relu(-tar_outputs + max_), dim=0).detach())*F.logsigmoid(tar_outputs*10 - max_.detach())
					#result_loss += -fine_tune_score_pos.mean()
				if model_type == 'TeRo':
					
					y_pos = outputs[:, 0]
					y_neg = outputs[:, 1:]
					M = y_pos.size(0)
					y_pos = 110-y_pos
					y_neg = 110-y_neg
					y_neg = y_neg.view(49, -1).transpose(0, 1)
					p = F.softmax(0.5*y_neg)
					loss_pos = torch.sum(F.softplus(-1 * y_pos))
					loss_neg = torch.sum(p * F.softplus(y_neg))
					result_loss = (loss_pos + loss_neg) / 2 / M
					
			
		return result_loss

class BoosterLoss(nn.Module):
	def __init__(self):
		super(BoosterLoss,self).__init__()
		bce_loss = nn.BCELoss()
		self.leaky_relu = nn.LeakyReLU(0.1)
	
	def masked_softmax(self, f, dim):
		f[f == 0] = -1000
		score = F.softmax(f, dim=1)
		return score

	def forward(self, can_outputs, can_target, type_):
		can_target = can_target.reshape(-1,1)
		if type_ == 'HyTE':
			result_loss = 0
			can_log = F.log_softmax(can_outputs, dim=1)
			can_log = can_log.gather(1, can_target)
			#result_loss = -1*can_log
			#result_loss = 0.1*result_loss.mean(dim=0)
			
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			result_loss += -0.1*fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			#result_loss += -fine_tune_score_pos.mean()
		
		if type_ == 'HyTE_ya':
			result_loss = 0
			can_log = F.log_softmax(can_outputs, dim=1)
			can_log = can_log.gather(1, can_target)
			result_loss = -1*can_log
			result_loss = result_loss.mean(dim=0)
			
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			result_loss += -fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			result_loss += -fine_tune_score_pos.mean()
		
		if type_ == 'TNT':
			can_log = F.log_softmax(can_outputs/5, dim=1)
			can_log = can_log.gather(1, can_target)
			result_loss = -1*can_log
			result_loss = 0.1*result_loss.mean(dim=0)
			
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			result_loss += -fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			result_loss += -fine_tune_score_pos.mean()
		if type_ == 'DE':
			result_loss = 0
			pos_outputs = can_outputs.gather(1, can_target)
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			result_loss += -fine_tune_score_pos.mean()

			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			result_loss += -fine_tune_score_neg.mean()
		if type_ == 'TA':
			result_loss = 0
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			#result_loss += -fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			result_loss += -fine_tune_score_pos.mean()

		if type_ == 'Timeplex':
			result_loss = 0
			can_log = F.log_softmax(can_outputs/5, dim=1)
			can_log = can_log.gather(1, can_target)
			#result_loss = -1*can_log
			#result_loss = 0.1*result_loss.mean(dim=0)
			
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(can_outputs, dim=1).detach()*F.logsigmoid(-can_outputs/10)).sum(dim = 1)
			#result_loss += -fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			result_loss += -fine_tune_score_pos.mean()

		if type_ == 'TeRo':
			'''
			y_pos = can_outputs[:, 0]
			y_neg = can_outputs[:, 1:]
			M = y_pos.size(0)
			y_pos = 110-y_pos
			y_neg = 110-y_neg
			y_neg = y_neg.view(49, -1).transpose(0, 1)
			p = F.softmax(0.5*y_neg)
			loss_pos = torch.sum(F.softplus(-1 * y_pos))
			loss_neg = torch.sum(p * F.softplus(y_neg))
			result_loss = (loss_pos + loss_neg) / 2 / M
			'''
			
			result_loss = 0
			'''
			can_log = F.log_softmax(can_outputs/5, dim=1)
			can_log = can_log.gather(1, can_target)
			result_loss = -1*can_log
			result_loss = 0.1*result_loss.mean(dim=0)
			'''
			y_neg = can_outputs[:, 1:]
			pos_outputs = can_outputs.gather(1, can_target)
			fine_tune_score_neg = (F.softmax(-y_neg, dim=1).detach()*F.softplus(-y_neg)).sum(dim = 1)
			#result_loss += 0.1*fine_tune_score_neg.mean()
			
			max_ = torch.max(can_outputs,dim=1)[0]
			fine_tune_score_pos = (1+self.masked_softmax(torch.relu(-pos_outputs + max_), dim=0).detach())*F.logsigmoid(pos_outputs*10 - max_.detach())
			#result_loss += -fine_tune_score_pos.mean()
			
		return result_loss

		
class HingeLoss(nn.Module):
	def __init__(self):
		super(HingeLoss,self).__init__()

	def forward(self, pos, neg):
		result_loss = torch.max(pos-neg+1, floatTensor([0.0]))
		result_loss = result_loss.mean()

		return result_loss

def orthogonalLoss(rel_embeddings, norm_embeddings):
	return torch.sum(torch.sum(torch.mm(rel_embeddings,norm_embeddings), dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
	norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
	return torch.mean(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))

def F_norm(martix):
	norm = torch.norm(martix)
	return torch.mean(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))

def get_cost(list_A, list_B):

	return 1-float(float(sum(list_A*list_B))/float(min(sum(list_A), sum(list_B))+1))
 
def dtw_measure(s, t):
	# slow
	#distance, path = fastdtw(np.array(s), np.array(t), dist=get_cost)
	
	# fast
	s_collected = np.array([0]*len(s[0]))
	for s_sample in s:
		s_collected = s_collected|s_sample

	t_collected = np.array([0]*len(t[0]))
	for t_sample in t:
		t_collected = t_collected|t_sample
	
	distance = 1-float(float(sum(s_collected*t_collected))/float(min(sum(s_collected), sum(t_collected))+1))
	
	return distance
