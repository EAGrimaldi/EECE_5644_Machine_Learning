import numpy as np
from scipy.stats import random_correlation
import matplotlib.pyplot as pp
import os.path

class Q2_true:
    def __init__(self, dimension, a, mu, sigma, alpha):
        #initializes a very simple GMM with 4 components on the corners of a square and about 10-20% overlap
        self.dimension = dimension
        self.a = a
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.identity = np.identity(dimension)
        self.z_cov = self.alpha*self.identity
    def draw_samples(self, N, K=10):
        X = np.random.multivariate_normal(self.mu, self.sigma, N)
        Z = np.random.multivariate_normal(np.zeros(self.dimension), self.z_cov, N)
        NoisyX = np.add(X, Z)
        scalar_product = np.dot(NoisyX, self.a)
        V = np.random.default_rng().normal(0.0,1.0,N)
        Y = np.add(scalar_product, V)
        if K!=0:
            XK = np.array_split(X, K, axis=0)
            YK = np.array_split(Y, K, axis=0)
            VK = np.array_split(V, K, axis=0)
            XnotK = []
            YnotK = []
            VnotK = []
            for k in range(K):
                tempXnotK = []
                tempYnotK = []
                tempVnotK = []
                for j in range(K):
                    if j!=k:
                        tempXnotK.append(XK[j])
                        tempYnotK.append(YK[j])
                        tempVnotK.append(YK[j])
                XnotK.append(np.concatenate(tempXnotK, axis=0))
                YnotK.append(np.concatenate(tempYnotK, axis=0))
                VnotK.append(np.concatenate(tempVnotK, axis=0))
            return [N, X, Y, V, K, XK, YK, VK, XnotK, YnotK, VnotK]
        else:
            return [N, X, Y, V, K]

class Q2_dataset:
    def __init__(self, stuff):
        self.dimension = 7
        self.N = stuff[0]
        self.X = stuff[1]
        self.Y = stuff[2]
        self.V = stuff[3]
        self.K = stuff[4]
        if self.K!=0:
            self.XK = stuff[5]
            self.YK = stuff[6]
            self.VK = stuff[7]
            self.XnotK = stuff[8]
            self.YnotK = stuff[9]
            self.VnotK = stuff[10]

def get_w(beta, x, y):
    (N, d) = x.shape
    z = np.concatenate((np.array([np.ones(N)]).T,x), axis=1)
    sum_y_z = np.dot(y, z)
    weird_I = (1/beta)*np.identity(d+1)
    sum_z_zT = np.zeros((d+1,d+1))
    for i in range(N):
        sum_z_zT += np.outer(z[i], z[i].T)
    weird_mat = weird_I + sum_z_zT
    w = np.dot(np.linalg.inv(weird_mat), sum_y_z)
    return w

def evaluate_sample(beta, w, z, y, v):
    neg2log_scale = np.log(2*np.pi)
    neg2log_exp = (y-np.dot(w.T, z))**2
    neg2log = neg2log_scale+neg2log_exp
    y_estimate = np.dot(w.T, z)+v
    error = abs(y_estimate-y)/y #percent error
    return neg2log, error, y_estimate

def evaluate_dataset(beta, w, x, y, v):
    N = y.size
    z = np.concatenate((np.array([np.ones(N)]).T,x), axis=1)
    neg2log = 0
    sum_error = 0
    temp_y = []
    for i in range(N):
        new_neg2log, new_sq_error, new_y = evaluate_sample(beta, w, z[i], y[i], v[i])
        neg2log += new_neg2log
        sum_error += new_sq_error
        temp_y.append(new_y)
    mean_error = sum_error/N
    y_guesses = np.array(temp_y)
    return neg2log, mean_error, y_guesses

def main():
    #generate data
    K = 10
    dimension = 7
    a = np.array([4,3,2,1,2,3,4])
    mu = np.array([1,2,3,4,3,2,1])
    file_path = 'rand_sigma.npy'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            sigma = np.load(fp)
            print(sigma)
    else:
        sigma = random_correlation.rvs((0.3,0.5,0.9,1.1,1.3,1.5,1.4))
        with open(file_path, 'wb') as fp:
            np.save(fp, sigma)
    base_alpha = np.trace(sigma)/dimension
    simple_alpha_space = [base_alpha/1000, base_alpha/100, base_alpha/10, base_alpha, base_alpha*10, base_alpha*100, base_alpha*1000]
    full_alpha_space = base_alpha*np.logspace(-3, 3, num=100)
    full_beta_choice = []
    full_neg2log = []
    full_mean_error = []
    for alpha in full_alpha_space:
        true = Q2_true(dimension, a, mu, sigma, alpha)
        Dtrain = Q2_dataset(true.draw_samples(100, K=K))
        Dtest = Q2_dataset(true.draw_samples(10000, K=0))
        beta_space = np.logspace(-3, 3, num=100)
        average_neg2log = []
        average_mean_error = []
        for i in range(100):
            beta = beta_space[i]
            tempsum_neg2log = 0
            tempsum_mse = 0
            for k in range(K):
                Xtrn = Dtrain.XnotK[k]
                Ytrn = Dtrain.YnotK[k]
                Xval = Dtrain.XK[k]
                Yval = Dtrain.YK[k]
                Vval = Dtrain.VK[k]
                w = get_w(beta, Xtrn, Ytrn)
                neg2log, mse, Y_guesses = evaluate_dataset(beta, w, Xval, Yval, Vval)
                tempsum_neg2log += neg2log
                tempsum_mse += mse
            average_neg2log.append(tempsum_neg2log/K)
            average_mean_error.append(tempsum_mse/K)
        beta_choice = beta_space[np.argmin(average_neg2log)]
        w_choice = get_w(beta_choice, Dtest.X, Dtest.Y)
        neg2log, mean_error, Y_guesses = evaluate_dataset(beta_choice, w, Dtest.X, Dtest.Y, Dtest.V)
        full_beta_choice.append(beta_choice)
        full_neg2log.append(neg2log)
        full_mean_error.append(mean_error)
        print("alpha %f" %alpha)
        print("beta: %f" %beta_choice)
        print("w0: %f" %w_choice[0])
        print("w: [%f %f %f %f %f %f %f]" %(w_choice[1],w_choice[2],w_choice[3],w_choice[4],w_choice[5],w_choice[6],w_choice[7]))
        print("-2log(p(D|w)): %f" %neg2log)
        print("mean error: %f" %mean_error)
        #pp.plot(np.linspace(0,Dtest.N-1,num=Dtest.N), Dtest.Y, 'r')
        #pp.plot(np.linspace(0,Dtest.N-1,num=Dtest.N), Y_guesses, 'g')
        #pp.show()
    pp.clf()
    pp.plot(full_alpha_space, full_beta_choice)
    pp.xscale("log")
    pp.yscale("log")
    pp.xlabel("alpha")
    pp.ylabel("beta")
    pp.show()
    pp.clf()
    pp.plot(full_alpha_space, full_neg2log)
    pp.xscale("log")
    pp.yscale("log")
    pp.xlabel("alpha")
    pp.ylabel("-2log(p(D|w))")
    pp.show()

if __name__ == "__main__":
    main()
