import pandas as pd
import numpy as np
import math

class GivenProbability:

	def __init__(self, df, find, variables):
		
		self.df = df
		self.variables = variables
		self.prob_lists = []
		self.find = find

	def createLists(self):

		for item in self.variables:

			temp_prob= self.df[item].unique()
			temp_dict = {}

			for i in temp_prob:
			    df_temp = self.df[self.df[item] == i]
			    prob = df_temp[self.find].value_counts(normalize=True)[1]
			    temp_dict[i] = prob
			
			self.prob_lists.append(temp_dict)

	def calculateProbability(self, givens):

		prob_for_givens = []

		self.createLists()

		if len(givens) != len(self.variables):
			print('Incorrect number of inputs')

		else:

			for i in range(len(givens)):

				prob_for_givens.append(self.prob_lists[i][givens[i]])

		return np.product(prob_for_givens)

