import numpy as np
import cvxopt
from cvxopt import matrix, spmatrix, solvers

solvers.options['show_progress'] = False

# Learning Non-Linear Combinations of Kernels
class NLK():

	def __init__(self, reg=0.001, tol=1e-9, degree=2, method='svm'):
		self.reg = reg
		self.tol = tol
		self.method = method
		self.degree = degree


	def svm_step(self, K_all, y, w):
		nb_samples = len(y)
		C = 1 / ( 2 * self.reg * nb_samples)

		nb_samples = len(y)
		r = np.arange(nb_samples)
		o = np.ones(nb_samples)
		z = np.zeros(nb_samples)

		K = np.sum((K_all * w[:, None, None]), axis=0) ** self.degree

		# see p 155/639 of the lecture
		P = matrix(K.astype(float), tc='d')
		q = matrix(-y, tc='d')
		G = spmatrix(np.r_[y, -y], np.r_[r, r + nb_samples], np.r_[r, r], tc='d')
		h = matrix(np.r_[o * C, z], tc='d')

		# call the solver
		sol = solvers.qp(P, q, G, h)

		# alpha
		a = np.ravel(sol['x'])
			
		return a


	def grad_step(self, K_all, w, alpha):
		K_t = np.sum(K_all * w[:, None, None], axis=0) ** (self.degree-1)
		
		grad = np.zeros(len(K_all))
		for m in range(len(K_all)):
			grad[m] = alpha.T.dot((K_t * K_all[m])).dot(alpha)
		
		return - self.degree * grad

	def normalize(self, u, norm):
		if norm == "l1":
			return u / np.sum(u)
		elif norm == "l2":
			return u / np.sqrt(np.sum(u**2))
		else:
			raise Exception('method not implemented. Choose between l1 and l2')


	def project(self, u, u0, delta, norm):
		u_hat = (u - u0)
		u_hat = np.abs(self.normalize(u_hat, norm) * delta)
		return u_hat + u0


	# K_train, K_all, Ytr_train ?
	def svm(self, K_all, y, u0=0, delta=1, norm='l2', n_iter=20, step=1):
		w = np.random.normal(0, 1, len(K_all)) / len(K_all)
		w = self.project(w, u0, delta, norm)
		new_w = 0

		score_prev = np.inf

		for _ in range(n_iter):
			alpha = self.svm_step(K_all, y, w)
			g = self.grad_step(K_all, w, alpha)

			new_w = w - step * g
			new_w = self.project(new_w, u0, delta, norm)

			score = np.linalg.norm(new_w - w, np.inf)

			if score > score_prev:
				step *= 0.8

			if score < self.tol:
				return new_w

			w = new_w
			score_prev = score.copy()

		return new_w


	def krr(self, K_all, y, u0=0, delta=1, norm='l2', n_iter=20, step=1):
		_, N, D = K_all.shape
		I = np.eye(N, D)

		w = np.random.normal(0, 1, len(K_all)) / len(K_all)
		w = self.project(w, u0, delta, norm)
		new_w = 0
		score_prev = np.inf

		for _ in range(n_iter):
			K_weighted = np.sum((K_all * w[:, None, None]), axis=0) ** self.degree
			alpha = np.linalg.inv(K_weighted + self.reg * I).dot(y)

			alpha = self.svm_step(K_all, y, w)
			g = self.grad_step(K_all, w, alpha)

			new_w = w - step * g
			new_w = self.project(new_w, u0, delta, norm)

			score = np.linalg.norm(new_w - w, np.inf)

			if score > score_prev:
				step *= 0.8

			if score < self.tol:
				return new_w

			w = new_w
			score_prev = score.copy()

		return new_w


	def run(self, K_all, y, u0=0, delta=1, norm='l2', n_iter=20, step=1):
		if self.method == 'krr':
			return self.krr(K_all, y, u0, delta, norm, n_iter, step)
		elif self.method == 'svm':
			return self.svm(K_all, y, u0, delta, norm, n_iter, step)
		else:
			raise Exception('method not available. Choose between krr and svm')
