import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from .condGMM import CondGMM


class CondDPGMM(CondGMM):
	"""
	Class that allows to define a conditional GMM starting from a DP GMM, as implemented by scikit-learn in BayesianGaussianMixture
	"""
	def __init__(self, fixed_indices, data, **DPGMM_kwargs):
		self.DPGMM = BayesianGaussianMixture(**DPGMM_kwargs).fit(data)
		super().__init__(self.DPGMM.weights_, self.DPGMM.means_, self.DPGMM.covariances_, fixed_indices)
