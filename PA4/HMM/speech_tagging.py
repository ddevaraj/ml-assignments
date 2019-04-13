import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	pi = [0]* len(tags)
	state_dict = {c : i for i,c in enumerate(tags)} # finding state dict

	# finding observation dict
	word_set = set()
	for i in range(len(train_data)):
		for word in train_data[i].words:
			word_set.add(word)
	obs_dict = {c : i for i,c in enumerate(word_set)}

	# print(obs_dict)
	# Finding initial prob - pi
	for i in range(len(train_data)):
		first_tag = (train_data[i].tags[0])
		pi[state_dict[first_tag]] += 1
	pi = np.divide(np.asarray(pi), len(train_data))
	# print(pi,len(pi),len(tags))

	# Finding transition matrix
	transition_matrix = np.zeros([len(tags), len(tags)])
	for i in range(len(train_data)):
		for j in range((train_data[i].length)-1):
			state1 = state_dict[train_data[i].tags[j]]
			state2 = state_dict[train_data[i].tags[j+1]]
			transition_matrix[state1][state2] += 1
			# print(train_data[i].tags[j], train_data[i].tags[j+1])

	# print(sum(transition_matrix[tag_index['NOUN']]))
	# print((transition_matrix[tag_index['NOUN']][tag_index['ADJ']]))
	# (e.T / e.sum(axis=1)).T
	trans_sum = []
	for i in range(len(tags)):
		row = sum(transition_matrix[i])
		if not row:
			row = 1
		trans_sum.append(row)
	trans_sum = np.asarray(trans_sum)
	transition_matrix = transition_matrix / trans_sum[:, None]
	transition_matrix[np.isnan(transition_matrix)] = 0


	# print('ho', (transition_matrix[state_dict['NOUN']][state_dict['ADJ']]))

	# Finding observation matrix
	obs_matrix = np.zeros([len(tags), len(word_set)])
	for i in range(len(train_data)):
		for j in range(len(train_data[i].words)):
			state1 = state_dict[train_data[i].tags[j]]
			state2 = obs_dict[train_data[i].words[j]]
			obs_matrix[state1][state2] += 1
	obs_sum = []
	for i in range(len(tags)):
		# print(sum(obs_matrix[i]))
		row = sum(obs_matrix[i])
		if not row:
			row = 1
		obs_sum.append(row)
	obs_sum = np.asarray(obs_sum)
	obs_matrix = obs_matrix / obs_sum[:, None]
	obs_matrix[np.isnan(obs_matrix)] = 0

	# print(obs_matrix[state_dict['NOUN']][obs_dict['time']])
	model = HMM(pi, transition_matrix, obs_matrix, obs_dict, state_dict)


	###################################################
	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	return tagging

