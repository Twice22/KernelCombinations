import numpy as np
import cvxopt

from scipy.optimize import fmin_l_bfgs_b
from cvxopt import matrix, spmatrix, solvers, spdiag


solvers.options['show_progress'] = False

def svm(K, y, C):
	nb_samples = len(y)
	r = np.arange(nb_samples)
	o = np.ones(nb_samples)
	z = np.zeros(nb_samples)

	# see p 155/639 of the lecture
	P = matrix(K.astype(float), tc='d')
	q = matrix(-y, tc='d')
	G = spmatrix(np.r_[y, -y], np.r_[r, r + nb_samples], np.r_[r, r], tc='d')
	h = matrix(np.r_[o * C, z], tc='d')

	# call the solver
	sol = solvers.qp(P, q, G, h)

	# alpha
	a = np.ravel(sol['x']) * y
	obj = - sol['primal objective']
		
	return a, obj


# TODO: change the linesearch function ?!
class SimpleMKL():

	def __init__(self, C=0.25, dual_threshold=1e-2, ratio_ls=1e-1, activate_thresholding=True, weight_precision=1e-8, verbose=False):
		self.C = C
		self.dual_threshold = dual_threshold
		self.activate_thresholding = activate_thresholding
		self.weight_precision = weight_precision
		self.verbose = verbose
		self.ratio_ls = ratio_ls


	def get_J(self, K, y, C):
		alpha_val, obj_val = svm(K, y, C)
		return alpha_val, obj_val

	# see equation 13 (naive way seems to be faster ?!)
	def get_dJ(self, kernels, Y, alpha):
		alpha_bis = np.outer(alpha, alpha)[None, :, :]
		Y_bis = Y[None, :, :]
	
		return - 0.5 * np.sum((alpha_bis * Y_bis) * kernels, axis=(1,2))


	# get descent direction (see equation 14)
	# actually we use the definition of Gradient_red just above equation 14
	# it is more intuitive to implement it using numpy
	def get_D(self, d, dJ, mu):
		dJ_red = - dJ + dJ[mu]
		
		# get all d_m such that d_m = 0 (we use d <= 0 as d must satisfy d >= 0)
		# and dJ[d_m] - dJ[mu] > 0
		pos_diff_grad = np.intersect1d(np.where(d <= 0)[0] , np.where(dJ_red < 0)[0])
		dJ_red[pos_diff_grad] = 0

		dJ_red[mu] = -np.sum(dJ_red)
		
		return dJ_red

	def get_gmax(self, d, D):
		idx_neg = np.where(D < 0)[0]
		if len(idx_neg):
			quotient = - d[idx_neg] / D[idx_neg]
			return np.min(quotient)
		
		return 0


	# line search from the matlab code: mklsvmupdate.m from the paper
	def line_search_paper(self, stepmin, stepmax, costmin, costmax, d, D, kernels, J_prev, Y, alpha0, bounds):
		"""
			costmin = J(d*)
			costmax = J_cross = J(d* + gamma_max * D)
			ratio_ls: is the maximum relative gap between stepmin and stpemax i.e between gamma_max and the gamma that we update
		"""
		gold = (np.sqrt(5) + 1) / 2

		step = np.array([stepmin, stepmax])
		cost = np.array([costmin, costmin])
		coord = np.argmin(cost)

		alpha = alpha0.copy()
		delta_max = stepmax

		min_precision = np.finfo(float).eps

		while (stepmax - stepmin) > self.ratio_ls * (abs(delta_max)) and stepmax > min_precision:
			stepmedr = stepmin + (stepmax - stepmin) / gold
			stepmedl = stepmin + (stepmedr - stepmin) / gold

			tmpmedr = d + stepmedr * D
			alpha_medr, costmedr = self.get_J_faster(np.sum(tmpmedr[:, None, None] * kernels, axis=0), Y, alpha0, bounds)

			tmpmedl = d + stepmedl * D
			alpha_medl, costmedl = self.get_J_faster(np.sum(tmpmedl[:, None, None] * kernels, axis=0), Y, alpha0, bounds)

			step = np.array([stepmin, stepmedl, stepmedr, stepmax])
			cost = np.array([costmin, costmedl, costmedr, costmax])
			coord = np.argmin(cost)

			if coord == 0:
				stepmax = stepmedl
				costmax = costmedl
				alpha = alpha_medl
			elif coord == 1:
				stepmax = stepmedr
				costmax = costmedr
				alpha = alpha_medr
			elif coord == 2:
				stepmin = stepmedl
				costmin = costmedl
				alpha = alpha_medl
			elif coord == 4:
				stepmin = stepmedr
				costmin = costmedr
				alpha = alpha_medr

		costNew = cost[coord]
		step = step[coord]
		if costNew < J_prev:
			return step, alpha, costNew
		else:
			return stepmin, alpha, costmin


	# usual linesearch
	def line_search(self, kernels, d, y, alpha0, bounds, gamma_max, J_cross, D, dJ, alpha=0.5, beta=0.99):
		# seek gamma such that:
		# f(x + gamma * dx) <= f(x) + alpha * gamma * df.dot(dx)
		# here it means:
		# J(d + gamma * D) <= J(d) + alpha * gamma * dJ.dot(D)
		gamma = gamma_max
		h = D.T.dot(dJ)
		
		while True:
			d_bis = (d + gamma * D)[:, None, None]
			K_weighted = np.sum(d_bis * kernels, axis=0)
			
			# use Newton instead of SVM for efficiency issue
			# we have the right to use scipy... So that should be OK ?!
			_, J_val = self.get_J_faster(K_weighted, y, alpha0, bounds)
			
			# add gamma < 1e-5 to break from the loop if gamma is to small
			if (J_val <= J_cross + gamma * alpha * h) or gamma < 1e-5:
				return gamma
			else:
				gamma *= beta
		
		return gamma

	def get_J_faster(self, K, Y, alpha0, bounds):
		n = K.shape[0]

		def dual(alpha):
			return 1/2 * alpha.T.dot(np.multiply(K, Y)).dot(alpha) - np.sum(alpha)

		def grad(alpha):
			"Gradient w.r.t alpha"
			return np.multiply(K, Y).dot(alpha) - np.ones(n) 

		alpha_val, obj_val, _ = fmin_l_bfgs_b(dual, alpha0, fprime=grad,
									bounds=bounds)
		obj_val *= -1

		return alpha_val, obj_val

	def threshold_precision(self, d, tol):
		res = d.copy() #new_d
		
		# put to 0 d_m that are very close to 0
		res[np.where(d < tol)[0]]=0
		
		# normalize (\sum_{m} d_m = 1)
		res = res / np.sum(res)
		
		return res

	def update_D(self, d, D, mu, tol):
		# get index of of d weights near 0
		# get index of reduce gradient below 0
		neg_idx = np.intersect1d(np.where(D <= 0)[0], np.where(d <= tol)[0])
		D[neg_idx] = 0

		if mu == 0:
			D[mu] = -np.sum(D[mu + 1:])
		else:
			D[mu] = -np.sum(np.concatenate((D[0:mu],D[mu + 1:]) ,0))
			
		return D


	def run(self, kernels, y):
		M = len(kernels) # number of kernels
		n = len(y) # number of samples
		
		# initialize the weight of each kernel to 1/M
		d = np.ones(M) / M
		
		# reduced gradient descent direction
		D = np.ones(M)
		
		dJ = None
		
		Y = np.outer(y, y)
		
		# needed for fmin_l_bfgs_b for scipy to speed up to algorithm
		# I could have use get_J in the linesearch but do several SVM
		# in a loop as a cost!
		bounds = [[0, self.C]] * n
		n_iter = 0
		
		while True:
			n_iter += 1
			
			if self.verbose:
				print('iteration %d, weights: %s' % (n_iter, d))
			
			
			# compute K_weighted = \sum_{m=1}^M d_m K_m
			K_weighted = np.sum(d[:, None, None] * kernels, axis=0)       
			alpha, J = self.get_J(K_weighted, y, self.C)
			
			# compute dJ_dm \forall m \in \{1, M\}
			dJ = self.get_dJ(kernels, Y, alpha)
			mu = np.argmax(d)
			
			# compute D: descent direction (see equation 14 of paper)
			D = self.get_D(d, dJ, mu)
			
			J_cross, d_cross, D_cross = 0, d.copy(), D.copy()
			J_prev = J
			
			n_inner_iter = 1
			
			# descent direction update
			# J_cross = J(d + gamma_max * D) and J = J(d)
			while J_cross < J:
				d, D = d_cross.copy(), D_cross.copy()
				
				if n_inner_iter > 1: # update J
					J = J_cross
				
				# gamma_max = min -dm/Dm on {m| Dm < 0}
				gamma_max = self.get_gmax(d, D) # does v = 0 makes any sense ?
				d_cross = d + gamma_max * D
		
				
				# compute J_cross by using an SVM solver with K = \sum_{m} d_cross[m] Km
				K_weighted_cross = np.sum(d_cross[:, None, None] * kernels, axis=0)
				alpha_cross, J_cross = self.get_J(K_weighted_cross, y, self.C)
				
				if J_cross < J:
					if self.verbose:
						print("inner iteration: weights :", d)
					D_cross = self.update_D(d_cross, D, mu, self.weight_precision)
					n_inner_iter += 1
			
			# Do a line searcg along D for gamma \in [0, \gamma_max]
			# get pt that minimizes J between d and d_cross
			# calls an LGBS algorithm to solve for each \gamma trial value
			#gamma = self.line_search(kernels, d, y, alpha, bounds, gamma_max, J_cross, D, dJ)

			# line search from matlab code from the original paper
			gamma, alpha, J = self.line_search_paper(0, gamma_max, J, J_cross,
													 d, D, kernels, J_prev, Y, alpha,
													  bounds)


			d += gamma * D
			if self.activate_thresholding:
				d = self.threshold_precision(d, self.weight_precision)
			
			# compute duality gap
			dJ_outer = self.get_dJ(kernels, Y, alpha)
			
			# stopping criterion: a RELATIVE duality gap below 0.01 (see p23)
			duality_gap = (J + np.max(-dJ_outer) - np.sum(alpha)) / J
			
			if self.verbose:
				print('duality gap: ',duality_gap)

			if duality_gap < self.dual_threshold:
				break
		
		return d