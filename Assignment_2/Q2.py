import numpy as np
import scipy.spatial.distance as sd
import sklearn.mixture as sklm
import matplotlib.pyplot as plt
import os.path as path
import time
import sys

class Q2_GMM:
    def __init__(self, num_components, test_mode=False, safe_mode=False, alpha=None, mu=None, sigma=None):
        if (test_mode):
            #initializes a very simple GMM with 4 components on the corners of a square and about 10-20% overlap
            self.C = 4
            self.alpha = [0.25, 0.25, 0.25, 0.25]
            self.mu = [np.array([1,1]), np.array([-1,1]), np.array([-1,-1]), np.array([1,-1])]
            var = 0.35
            self.sigma = [np.array([[var,0],[0,var]]), np.array([[var,0],[0,var]]), np.array([[var,0],[0,var]]), np.array([[var,0],[0,var]])]
        else:
            if (alpha is None) and (mu is None) and (sigma is None):
                self.C = num_components
                self.alpha = []
                self.mu = []
                self.sigma = []
                var_backup = []
                need_new_var = True
                if safe_mode:
                    if path.exists("var_backup_%d.npy" %self.C):
                        file_path = "var_backup_"+str(self.C)+".npy"
                        with open(file_path, "rb") as fp:
                            var_backup = np.load(fp)
                        need_new_var = False
                for c in range(self.C):
                    c=c #quiet pylint
                    self.alpha.append(1.0/float(self.C))
                    self.mu.append(np.array([c,c]))
                    if (not safe_mode):
                        var = 0.2+0.3*np.random.rand()
                    else:
                        if (need_new_var):
                            var = 0.2+0.3*np.random.rand()
                            var_backup.append(var)
                        if (not need_new_var):
                            var = var_backup[c]
                    self.sigma.append(np.array([[var,0],[0,var]]))
                if (safe_mode) and (need_new_var):
                    file_path = "var_backup_"+str(self.C)+".npy"
                    with open(file_path, "wb") as fp:
                        np.save(fp, np.array(var_backup))
            elif (alpha is not None) and (mu is not None) and (sigma is not None) and (len(alpha)==num_components) and (len(mu)==num_components) and (len(sigma)==num_components):
                self.C = num_components
                self.alpha = alpha
                self.mu = mu
                self.sigma = sigma
            else:
                sys.exit("something is not right...")
    def print_params(self):
        print("components: %d" %self.C)
        print("priors:")
        for c in range(self.C):
            print(self.alpha[c])
        print("means:")
        for c in range(self.C):
            print(self.mu[c])
        print("covariances:")
        for c in range(self.C):
            print(self.sigma[c])
    def draw_samples(self, num_samples, K=5, want_labels=False, want_dbl=False):
        tempData = []
        tempLabels = []
        tempDataByLabel = []
        tempDataByKfold = []
        tempNByKfold = []
        for c in range(self.C):
            c=c #quiet pylint
            tempDataByLabel.append([])
        for k in range(K):
            k=k #quiet pylint
            tempDataByKfold.append([])
            tempNByKfold.append(0)
        k=0
        for i in range(num_samples):
            i=i #quiet pylint
            rand_comp = np.random.randint(0,high=self.C)
            sample = np.random.multivariate_normal(self.mu[rand_comp], self.sigma[rand_comp])
            tempData.append(sample)
            tempLabels.append(rand_comp)
            tempDataByLabel[rand_comp].append(sample)
            tempDataByKfold[k].append(sample)
            tempNByKfold[k] += 1
            k += 1
            if (k >= K):
                k = 0
        Data = np.transpose(np.array(tempData))
        Labels = np.array(tempLabels)
        Data_by_Labels = []
        for c in range(self.C):
            Data_by_Labels.append(np.transpose(np.array(tempDataByLabel[c])))
        Data_by_Kfold = []
        for k in range(K):
            Data_by_Kfold.append(np.transpose(np.array(tempDataByKfold[k])))
        N_by_Kfold = tempNByKfold
        return data_set(num_samples, Data, Labels, Data_by_Labels, Data_by_Kfold, N_by_Kfold, K)

class data_set:
    def __init__(self, num_samples, Data, Labels, Data_by_Labels, Data_by_Kfold, N_by_Kfold, num_partitions):
        self.N = num_samples
        self.D = Data
        self.L = Labels
        self.DBL = Data_by_Labels
        self.DBK_val = Data_by_Kfold
        self.DBK_trn = []
        self.NBK_val = N_by_Kfold
        self.NBK_trn = []
        self.K = num_partitions
        for k in range(self.K):
            trn_list = []
            tempsum = 0
            for l in range(self.K):
                if (l != k):
                    trn_list.append(self.DBK_val[l])
                    tempsum += self.NBK_val[l]
            self.DBK_trn.append(np.concatenate(trn_list, axis=1))
            self.NBK_trn.append(tempsum)
    def fit_GMM(self, C, D_trn=None, rand_inits=4, print_progess=False):
        if (D_trn is None):
            D_trn = self.D
        if print_progess:
            print("Beginning EM estimation of GMM with %d components..." %C)
            start_time = time.time()
        estimate = sklm.GaussianMixture(C, n_init=rand_inits).fit(np.transpose(D_trn)) #covariance_type='spherical', 
        best_alpha = estimate.weights_
        best_mu = estimate.means_
        best_sigma = estimate.covariances_
        if print_progess:
            print("EM estimation complete")
            end_time = time.time()
            loop_hours, rem = divmod(end_time - start_time, 3600)
            loop_mins, loop_secs = divmod(rem, 60)
            print("Elapsed Time: %d:%d:%d" %(loop_hours, loop_mins, loop_secs))
        self.estimate = Q2_GMM(C, alpha=best_alpha, mu=best_mu, sigma=best_sigma)
        return self.estimate
    def EM_estimate_for_GMM_with_C_components(self, C, rand_inits=4):
        #Implementing this was a good learning experience, but unfortunately it is unreasonably slow
        #Thankfully sklearn.mixture has tools to accomplish this that utilize C++ for performance faster by several orders of magnitude
        print("Beginning EM estimation of GMM with %d components..." %C)
        start_time = time.time()
        best_alpha = []
        best_mu = []
        best_sigma = []
        for c in range(C):
            c=c #quiet pylint
            best_alpha.append(None)
            best_mu.append(None)
            best_sigma.append(None)
        best_LLH = None
        best_r = None
        D = self.D #training data
        N = self.N #number of samples in D
        for r in range(rand_inits):
            r=r #quiet pylint
            alpha = []
            mu = []
            lump = []
            sigma = []
            empty_lumps = True
            while (empty_lumps):
                for c in range(C):
                    alpha.append([(1.0/float(C))])
                    mu.append([D[:,np.random.randint(0,N)]])
                    lump.append([])
                for i in range(N):
                    tempdist = []
                    for c in range(C):
                        tempdist.append(sd.pdist(np.array([D[:,i], mu[c][0]])))
                    l = np.argmin(tempdist)
                    lump[l].append(D[:,i])
                empty_lumps = False
                for c in range(C):
                    if not lump[c]:
                       empty_lumps =  True
                    else:
                        sigma.append([np.cov(np.transpose(np.array(lump[c])))])
            converged = False
            i=0
            while(not converged):
                p_wish = []
                p_wish_total = []
                wish_weight = []
                new_alpha = []
                new_mu = []
                new_sigma = []
                for c in range(C):
                    p_wish.append([])
                    p_wish_total.append(0)
                    wish_weight.append([])
                    new_mu.append(np.array([0.0, 0.0]))
                    new_sigma.append(np.array([[0.0, 0.0],[0.0, 0.0]]))
                for j in range(N):
                    g_pdf = []
                    g_total = 0
                    for c in range(C):
                        g_pdf.append(self.compute_single_gaussian_likelihood(D[:,j], alpha[c][i], mu[c][i], sigma[c][i]))
                        g_total += g_pdf[c]
                    for c in range(C):
                        p_wish[c].append(g_pdf[c]/g_total)
                        p_wish_total[c] += p_wish[c][j]
                for c in range(C):
                    new_alpha.append(p_wish_total[c]/N)
                for j in range(N):
                    for c in range(C):
                        wish_weight[c].append(p_wish[c][j]/p_wish_total[c])
                        new_mu[c] += wish_weight[c][j]*D[:,j]
                for j in range(N):
                    for c in range(C):
                        new_sigma[c] += wish_weight[c][j]*np.outer(np.subtract(D[:,j], new_mu[c]), np.subtract(D[:,j], new_mu[c]))
                for c in range(C):
                    alpha[c].append(new_alpha[c])
                    mu[c].append(new_mu[c])
                    sigma[c].append(new_sigma[c])
                i += 1
                converged = self.check_convergence_for_GMM_with_C_components(C, i, alpha, mu, sigma)
            print("%d iterations to convergence on random init %d" %(i, r))
            curr_LLH = self.compute_LLH_for_EM_in_progress(C, i, alpha, mu, sigma)
            if (best_LLH == None):
                for c in range(C):
                    best_alpha[c] = alpha[c][i]
                    best_mu[c] = mu[c][i]
                    best_sigma[c] = sigma[c][i]
                best_LLH = curr_LLH
                best_r = r
            elif (curr_LLH > best_LLH):
                for c in range(C):
                    best_alpha[c] = alpha[c][i]
                    best_mu[c] = mu[c][i]
                    best_sigma[c] = sigma[c][i]
                best_LLH = curr_LLH
                best_r = r
        print("EM estimation complete")
        end_time = time.time()
        loop_hours, rem = divmod(end_time - start_time, 3600)
        loop_mins, loop_secs = divmod(rem, 60)
        print("Elapsed Time: %d:%d:%d" %(loop_hours, loop_mins, loop_secs))
        print("Best estimate from random init %d" %best_r)
        self.estimate = Q2_GMM(C, alpha=best_alpha, mu=best_mu, sigma=best_sigma)
        return self.estimate
    def compute_single_gaussian_likelihood(self, xi, alpha, mu, sigma):
        g_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, mu).T, np.linalg.inv(sigma)), np.subtract(xi, mu)))
        g_scale = ((2.0*np.pi)**(-1))*(np.linalg.det(sigma)**(-0.5))
        return (alpha*g_scale*g_exp)
    def compute_LLH_for_EM_in_progress(self, C, i, alpha, mu, sigma):
        LLH = 0
        D = self.D
        N = self.N
        for n in range(N):
            g_total = 0
            for c in range(C):
                g_total += self.compute_single_gaussian_likelihood(D[:,n], alpha[c][i], mu[c][i], sigma[c][i])
            LLH += np.log(g_total)
        return LLH
    def check_convergence_for_GMM_with_C_components(self, C, i, alpha, mu, sigma):
        d_alpha = []
        d_mu = []
        d_sigma = []
        for c in range(C):
            d_alpha.append(abs(alpha[c][i]-alpha[c][i-1]))
            d_mu.append(np.sum(np.absolute(np.subtract(mu[c][i], mu[c][i-1]))))
            d_sigma.append(np.sum(np.sum(np.absolute(np.subtract(sigma[c][i], sigma[c][i-1]))))) 
        d_params =  np.sum(d_alpha) + np.sum(d_mu) + np.sum(d_sigma)
        return (d_params<0.02)
    def compute_LLH_for_estimate(self, D=None, N=None, GMM_pdf=None):
        LLH = 0
        if (D is None) and (N is None):
            D = self.D
            N = self.N
        elif (D is None) and (N is not None):
            sys.exit("D and N do not match")
        elif (D is not None) and (N is None):
            sys.exit("D and N do not match")
        if (GMM_pdf is None):
            C = self.estimate.C
            alpha = self.estimate.alpha
            mu = self.estimate.mu
            sigma = self.estimate.sigma
        else:
            C = GMM_pdf.C
            alpha = GMM_pdf.alpha
            mu = GMM_pdf.mu
            sigma = GMM_pdf.sigma
        for n in range(N):
            g_total = 0
            for c in range(C):
                g_total += self.compute_single_gaussian_likelihood(D[:,n], alpha[c], mu[c], sigma[c])
            LLH += np.log(g_total)
        return LLH
    def estimate_GMM_order_BIC(self, show_curve=False):
        pdf_estimate = []
        BIC = []
        model_order_estimate = []
        num_samples = 2*self.N #due to data dimensionality of 2
        max_model_order = 15
        for m in range(max_model_order):
            C = m+1
            pdf_estimate.append(self.fit_GMM(C))
            num_params = 6*C-1 #generalizes to d dimensions as C-1 + C*d + C*(d + scipy.misc.comb(N, 2))
            BICtemp = (-2)*self.compute_LLH_for_estimate() + num_params*np.log(num_samples) 
            BIC.append(BICtemp)
            model_order_estimate.append(C)
        if show_curve:
            plt.clf()
            plt.plot(model_order_estimate, BIC, "r")
            plt.suptitle("BIC results")
            plt.show()
        i = np.argmin(BIC)
        best_estimate = pdf_estimate[i].C
        return best_estimate
    def estimate_GMM_order_Kfold(self, show_curve=False):
        pdf_estimate = []
        LLH = []
        ave_LLH = []
        model_order_estimate = []
        max_model_order = 15
        for m in range(max_model_order):
            C = m+1
            pdf_estimate.append([])
            LLH.append([])
            tempsum = 0
            for k in range(self.K):
                pdf_estimate[m].append(self.fit_GMM(C, D_trn=self.DBK_trn[k]))
                LLH[m].append(((-2)*self.compute_LLH_for_estimate(D=self.DBK_val[k], N=self.NBK_val[k]))/self.NBK_val[k])
                tempsum += LLH[m][k]
            ave_LLH.append(tempsum/self.K)
            model_order_estimate.append(C)
        best_estimate = np.argmin(ave_LLH)+1
        if show_curve:
            plt.clf()
            plt.plot(model_order_estimate, ave_LLH, "r")
            plt.suptitle("Kfold results")
            plt.show()
        return best_estimate

def main():
    #script start timer
    print("Beginning script...")
    main_start_time = time.time()
    #settings
    num_gaussians = 10 #req 10+
    data_power = [2,3,4,5] #req [2,3,4,5] append(6) if I have extra time
    num_experiments = 100 #req 100+
    #experiment
    true_pdf = Q2_GMM(num_gaussians, safe_mode=True)
    results_BIC = []
    results_Kfold = []
    old_temp_time = time.time()
    for p in range(len(data_power)):
        print("Beginning p=%d experiements" %(data_power[p]))
        results_BIC.append([])
        results_Kfold.append([])
        N = 10**data_power[p]
        for e in range(num_experiments):
            print(" %d" %e, end="\r")
            training_set = true_pdf.draw_samples(N)
            results_BIC[p].append(training_set.estimate_GMM_order_BIC())
            results_Kfold[p].append(training_set.estimate_GMM_order_Kfold())
        #save results
        file_path = "BIC_results_for_p="+str(data_power[p])+".npy"
        with open(file_path, "wb") as fp:
            np.save(fp, np.array(results_BIC[p]))
        file_path = "Kfold_results_for_p="+str(data_power[p])+".npy"
        with open(file_path, "wb") as fp:
            np.save(fp, np.array(results_Kfold[p]))
        #lap timer
        print("p=%d experiments complete" %(p+2))
        new_temp_time = time.time()
        temp_hours, temp_rem = divmod(new_temp_time - old_temp_time, 3600)
        temp_mins, temp_secs = divmod(temp_rem, 60)
        print("Elapsed Time: %d:%d:%d" %(temp_hours, temp_mins, temp_secs))
        old_temp_time = new_temp_time
    #script end timer
    print("Script complete")
    main_end_time = time.time()
    full_hours, remainder = divmod(main_end_time - main_start_time, 3600)
    full_mins, full_secs = divmod(remainder, 60)
    print("Elapsed Time: %d:%d:%d" %(full_hours, full_mins, full_secs))

if __name__ == "__main__":
    main()

