import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import math

class ProbabilityDensity:

	def __init__(self, data):
		self.data = np.reshape(data, (-1,1))
		self.scaled_data = None
		self.scaled_max = None
		self.scaler = None
		self.kde = None
		self.start = None
		self.end = None
		self.x_prob = None
		self.kde_prob = None
		self.int_prob = None
	
	def fit_pdf(self):
		# SCALE DATA WITH SCIKIT-LEARN STANDARD SCALER
		self.scaler = StandardScaler()

		self.scaled_data = self.scaler.fit_transform(self.data)

		if abs(self.scaled_data[0][0]) < abs(self.scaled_data[-1][0]):
			self.scaled_max = math.ceil(abs(self.scaled_data[0][0])) + 2
			print('max')
		else:
			self.scaled_max = math.ceil(abs(self.scaled_data[-1][0])) + 2

		# CREATE KERNAL DENSITY ESTIMATE MODEL
		self.kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(self.scaled_data)

	def view_hist(self):

		# CREATE EQUALLY SPACED X AXIS
		x_axis = np.linspace(-self.scaled_max,self.scaled_max, 1000)[:, np.newaxis]

		x_space = x_axis[1] - x_axis[0]

		# CONVERT X AXIS DENSITY TO PROBABILITY
		self.kde_prob = np.exp(self.kde.score_samples(x_axis))

		# PLOT HISTOGRAM AND MODEL
		plt.hist(self.scaled_data, bins=10, density=True)

		plt.plot(x_axis, self.kde_prob)

		plt.show()

	def find_prob(self, value, type):

		if type == '>':

			start = self.scaler.transform([[value]])
			self.start = start[0][0]
			self.end = self.scaled_max

		elif type == '<':

			self.start = -self.scaled_max
			end = self.scaler.transform([[value]])
			self.end = end[0][0]

		self.x_prob = np.linspace(self.start,self.end, 1000)[:, np.newaxis]

		x_space = self.x_prob[1] - self.x_prob[0]

		self.int_prob = np.exp(self.kde.score_samples(self.x_prob))

		interval_prob = np.trapz(self.int_prob, dx=x_space)

		return interval_prob

	def view_prob(self):

		# CREATE EQUALLY SPACED X AXIS
		x_axis = np.linspace(-self.scaled_max,self.scaled_max, 1000)[:, np.newaxis]

		x_space = x_axis[1] - x_axis[0]

		x_prob = self.x_prob.ravel().tolist()

		# PLOT HISTOGRAM AND MODEL
		plt.hist(self.scaled_data, bins=10, density=True, color='orange')

		plt.plot(x_axis, self.kde_prob, color='blue')
		plt.fill_between(x_prob, self.int_prob, color='red', alpha=.6)
		plt.axvline(x = self.end, color='red')
		plt.axvline(x = self.start, color='red')

		plt.show()

