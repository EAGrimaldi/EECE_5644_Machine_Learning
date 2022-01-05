import numpy as np
import scipy.spatial.distance as sd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
import sys

class Q1:
    def __init__(self, N):
        self.Dval_N = N
        self.true = self.pdf(0.6, 0.4, np.array([5,0]), np.array([0,4]), np.array([3,2]), np.array([[4,0],[0,2]]), np.array([[1,0],[0,3]]), np.array([[2,0],[0,2]]), 0.5, 0.5)
        (self.D100, self.D100_Labels, self.D100_0s, self.D100_1s, self.D100_LRTs, self.D100_LRTs_0s, self.D100_LRTs_1s) = self.generate_data(100)
        (self.D1k, self.D1k_Labels, self.D1k_0s, self.D1k_1s, self.D1k_LRTs, self.D1k_LRTs_0s, self.D1k_LRTs_1s) = self.generate_data(1000)
        (self.D10k, self.D10k_Labels, self.D10k_0s, self.D10k_1s, self.D10k_LRTs, self.D10k_LRTs_0s, self.D10k_LRTs_1s) = self.generate_data(10000)
        (self.Dval, self.Dval_Labels, self.Dval_0s, self.Dval_1s, self.Dval_LRTs, self.Dval_LRTs_0s, self.Dval_LRTs_1s) = self.generate_data(self.Dval_N)
    class pdf:
        def __init__(self, PL0, PL1, m01, m02, m1, C01, C02, C1, w1, w2):
            self.model = "GMM"
            self.PL0 = PL0
            self.PL1 = PL1
            self.m01 = m01
            self.m02 = m02
            self.m1 = m1
            self.C01 = C01
            self.C02 = C02
            self.C1 = C1
            self.w1 = w1
            self.w2 = w2
            self.theoretical_gamma = self.PL0/self.PL1
    def generate_data(self, N):
        tempData = []
        tempLabels = []
        tempData0s = []
        tempData1s = []
        tempLRTs = []
        tempLRTs0s = []
        tempLRTs1s = []
        for i in range(N):
            i=i #quiet pylint
            if np.random.rand() > self.true.PL1:
                tempLabels.append(0)
                if np.random.rand() > 0.5:
                    sample = np.random.multivariate_normal(self.true.m01, self.true.C01)
                else:
                    sample = np.random.multivariate_normal(self.true.m02, self.true.C02)
                tempData0s.append(sample)
                tempLRTs0s.append(self.compute_LRT(sample, self.true))
            else:
                tempLabels.append(1)
                sample = np.random.multivariate_normal(self.true.m1, self.true.C1)
                tempData1s.append(sample)
                tempLRTs1s.append(self.compute_LRT(sample, self.true))
            tempData.append(sample)
            tempLRTs.append(self.compute_LRT(sample, self.true))
        Data = np.transpose(np.array(tempData))
        Labels= np.array(tempLabels)
        D0s = np.transpose(np.array(tempData0s))
        D1s = np.transpose(np.array(tempData1s))
        LRTs = np.array(tempLRTs)
        LRTs0s = np.array(tempLRTs0s)
        LRTs1s = np.array(tempLRTs1s)
        return (Data, Labels, D0s, D1s, LRTs, LRTs0s, LRTs1s)
    def compute_LRT(self, xi, pdf):
        if (pdf.model=="GMM"):
            p1_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, pdf.m1).T, np.linalg.inv(pdf.C1)), np.subtract(xi, pdf.m1)))
            p1_det = np.linalg.det(pdf.C1)**(-0.5)
            p1 = p1_det*p1_exp
            p01_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, pdf.m01).T, np.linalg.inv(pdf.C01)), np.subtract(xi, pdf.m01)))
            p01_det = np.linalg.det(pdf.C01)**(-0.5)
            p02_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, pdf.m02).T, np.linalg.inv(pdf.C02)), np.subtract(xi, pdf.m02)))
            p02_det = np.linalg.det(pdf.C02)**(-0.5)
            p0 = pdf.w1*p01_det*p01_exp+pdf.w2*p02_det*p02_exp
            return p1/p0
        elif (pdf.model=="LLF"):
            h = pdf.eval_at(xi)
            return (h*pdf.PL0)/((1-h)*pdf.PL1)
        elif (pdf.model=="LQF"):
            h = pdf.eval_at(xi)
            return (h*pdf.PL0)/((1-h)*pdf.PL1)
        else:
            sys.exit("pdf model is not good")
    def generate_gammas(self):
        tempgammas = []
        tempLRTs = np.copy(self.Dval_LRTs)
        tempLRTs.sort()
        for i in range(self.Dval_N):
            if i == 0:
                tempgammas.append(0)
            else:
                tempgammas.append(tempLRTs[i-1]+((tempLRTs[i]-tempLRTs[i-1])/2.0))
        tempgammas.append(2.0*tempLRTs[self.Dval_N-1])
        self.gammas = np.array(tempgammas)
    def classify(self, LRT, gamma):
        if LRT > gamma:
            return 1
        elif LRT < gamma:
            return 0
        else:
            sys.exit("can't quite decide: %f >?< %f" %(LRT, gamma))
    def compute_confusion(self, gamma, pdf):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(self.Dval_N):
            classification = self.classify(self.compute_LRT(self.Dval[:,i], pdf), gamma)
            if (classification == 1):
                if (classification == self.Dval_Labels[i]):
                    TP += 1
                else:
                    FP += 1
            else:
                if (classification == self.Dval_Labels[i]):
                    TN += 1
                else:
                    FN += 1
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        TNR = TN/(FP+TN)
        FNR = FN/(TP+FN)
        return (FPR, TPR, FNR, TNR)
    def plot_ROC(self, pdf, fig_name):
        FPRlist = []
        TPRlist = []
        FNRlist = []
        TNRlist = []
        min_gamma = 0
        min_P_error = 1
        min_ROC_point = [1, 1, 0, 0]
        print("Beginning "+fig_name+"...")
        start_time = time.time()
        progress_bar = 0
        j=0
        for i in range(self.gammas.size):
            if (j==3):
                (FPR, TPR, FNR, TNR) = self.compute_confusion(self.gammas[i], pdf)
                FPRlist.append(FPR)
                TPRlist.append(TPR)
                FNRlist.append(FNR)
                TNRlist.append(TNR)
                P_error = FPR*pdf.PL0+FNR*pdf.PL1
                if P_error<min_P_error:
                    min_P_error = P_error
                    min_gamma = self.gammas[i]
                    min_ROC_point = (FPR, TPR, FNR, TNR)
                progress = int(100.0*i/(self.gammas.size-1))
                if (progress%5==0) and (progress>progress_bar):
                    print("%d%% TPR %f FPR %f gamma %f" %(progress, TPR, FPR, self.gammas[i])) #detailed progress
                    progress_bar += 5
            j += 1
            if (j==4):
                j=0
        print ("ROC Curve complete")
        end_time = time.time()
        loop_hours, rem = divmod(end_time - start_time, 3600)
        loop_mins, loop_secs = divmod(rem, 60)
        print("Elapsed Time: %d:%d:%d" %(loop_hours, loop_mins, loop_secs))
        print("gamma = %f achieves minimum P_error = %f" %(min_gamma, min_P_error))
        plt.clf()
        plt.plot(FPRlist, TPRlist, "b")
        plt.plot(min_ROC_point[0], min_ROC_point[1], "bo")
        plt.annotate("(%f, %f)\ngamma = %f\nP_error = %f"%(min_ROC_point[0], min_ROC_point[1], min_gamma, min_P_error), (min_ROC_point[0], min_ROC_point[1]), (.2,.6), arrowprops={"arrowstyle": "->"})
        plt.plot([1,0],[1,0], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.suptitle(fig_name)
        plt.savefig(fig_name)
        self.empirical_gamma = min_gamma
    def part1(self):
        self.generate_gammas()
        self.plot_ROC(self.true, "ROC Curve from True PDF")
        #self.plot_decision_boundary()
    def ML_estimate_pdf(self):
        print("Estimating PDF from %d samples..." %self.Dtrn_N)
        PL0 = self.Dtrn_LRTs_0s.size/self.Dtrn_N
        PL1 = self.Dtrn_LRTs_1s.size/self.Dtrn_N
        m1 = np.mean(self.Dtrn_1s, axis=1, dtype=np.float64)
        C1 = np.cov(self.Dtrn_1s)
        (m01, m02, C01, C02, w1, w2) = self.EM_estimate_for_GMM(10)
        print("PL0 = %f" %PL0)
        print("PL1 = %f" %PL1)
        print("m01 =")
        print(m01)
        print("m02 =")
        print(m02)
        print("m1 =")
        print(m1)
        print("C01 =")
        print(C01)
        print("C02 =")
        print(C02)
        print("C1 =")
        print(C1)
        print("w1 =  %f" %w1)
        print("w2 =  %f" %w2)
        self.ML_estimate = self.pdf(PL0, PL1, m01, m02, m1, C01, C02, C1, w1, w2)
    def EM_estimate_for_GMM(self, rand_inits=1):
        print("Beginning EM estimation...")
        start_time = time.time()
        best_m01 = None
        best_m02 = None
        best_C01 = None
        best_C02 = None
        best_w1 = None
        best_w2 = None
        best_LLH = None
        best_r = None
        N=self.Dtrn_LRTs_0s.size
        for r in range(rand_inits):
            r=r #quiet pylint
            m01 = [self.Dtrn_0s[:,np.random.randint(0,N)]]
            m02 = [self.Dtrn_0s[:,np.random.randint(0,N)]]
            lump1 = []
            lump2 = []
            for i in range(N):
                if sd.pdist(np.array([self.Dtrn_0s[:,i], m01[0]])) < sd.pdist(np.array([self.Dtrn_0s[:,i], m02[0]])):
                    lump1.append(self.Dtrn_0s[:,i])
                else:
                    lump2.append(self.Dtrn_0s[:,i])
            C01 = [np.cov(np.transpose(np.array(lump1)))]
            C02 = [np.cov(np.transpose(np.array(lump2)))]
            w1 = [0.5]
            w2 = [0.5]
            converged = False
            i=0
            while(not converged):
                p_wish_g1 = []
                p_wish_g2 = []
                p_wish_total_g1 = 0
                p_wish_total_g2 = 0
                for j in range(N):
                    g1_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(self.Dtrn_0s[:,j], m01[i]).T, np.linalg.inv(C01[i])), np.subtract(self.Dtrn_0s[:,j], m01[i])))
                    g2_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(self.Dtrn_0s[:,j], m02[i]).T, np.linalg.inv(C02[i])), np.subtract(self.Dtrn_0s[:,j], m02[i])))
                    g1_scale = ((2.0*np.pi)**(-1))*(np.linalg.det(C01[i])**(-0.5))
                    g2_scale = ((2.0*np.pi)**(-1))*(np.linalg.det(C02[i])**(-0.5))
                    g1 = w1[i]*g1_scale*g1_exp
                    g2 = w2[i]*g2_scale*g2_exp
                    p_wish_g1.append(g1/(g1+g2))
                    p_wish_g2.append(g2/(g1+g2))
                    p_wish_total_g1 += p_wish_g1[j]
                    p_wish_total_g2 += p_wish_g2[j]
                new_w1 = p_wish_total_g1/N
                new_w2 = p_wish_total_g2/N
                wish_weight_g1 = []
                wish_weight_g2 = []
                new_m01 = np.array([0.0, 0.0])
                new_m02 = np.array([0.0, 0.0])
                for j in range(N):
                    wish_weight_g1.append(p_wish_g1[j]/p_wish_total_g1)
                    wish_weight_g2.append(p_wish_g2[j]/p_wish_total_g2)
                    new_m01 += wish_weight_g1[j]*self.Dtrn_0s[:,j]
                    new_m02 += wish_weight_g2[j]*self.Dtrn_0s[:,j]
                new_C01 = np.array([[0.0, 0.0],[0.0, 0.0]])
                new_C02 = np.array([[0.0, 0.0],[0.0, 0.0]])
                for j in range(N):
                    new_C01 += wish_weight_g1[j]*np.outer(np.subtract(self.Dtrn_0s[:,j], new_m01), np.subtract(self.Dtrn_0s[:,j], new_m01))
                    new_C02 += wish_weight_g2[j]*np.outer(np.subtract(self.Dtrn_0s[:,j], new_m02), np.subtract(self.Dtrn_0s[:,j], new_m02))
                m01.append(new_m01)
                m02.append(new_m02)
                C01.append(new_C01)
                C02.append(new_C02)
                w1.append(new_w1)
                w2.append(new_w2)
                i += 1
                converged = self.check_convergence_for_GMM(i, m01, m02, C01, C02, w1, w2)
            print("%d iterations to convergence on random init %d" %(i, r))
            curr_LLH = self.compute_LLH_for_GMM(m01[i], m02[i], C01[i], C02[i], w1[i], w2[i])
            if (best_LLH == None):
                best_m01 = m01[i]
                best_m02 = m02[i]
                best_C01 = C01[i]
                best_C02 = C02[i]
                best_w1 = w1[i]
                best_w2 = w2[i]
                best_LLH = curr_LLH
                best_r = r
            elif (curr_LLH > best_LLH):
                best_m01 = m01[i]
                best_m02 = m02[i]
                best_C01 = C01[i]
                best_C02 = C02[i]
                best_w1 = w1[i]
                best_w2 = w2[i]
                best_LLH = curr_LLH
                best_r = r
        print("EM estimation complete")
        end_time = time.time()
        loop_hours, rem = divmod(end_time - start_time, 3600)
        loop_mins, loop_secs = divmod(rem, 60)
        print("Elapsed Time: %d:%d:%d" %(loop_hours, loop_mins, loop_secs))
        print("Best estimate from random init %d" %best_r)
        return best_m01, best_m02, best_C01, best_C02, best_w1, best_w2
    def compute_LLH_for_GMM(self, m01, m02, C01, C02, w1, w2):
        LLH = 0
        N = self.Dtrn_LRTs_0s.size
        for i in range(N):
            g1_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(self.Dtrn_0s[:,i], m01).T, np.linalg.inv(C01)), np.subtract(self.Dtrn_0s[:,i], m01)))
            g1_scale = ((2.0*np.pi)**(-1))*(np.linalg.det(C01)**(-0.5))
            g2_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(self.Dtrn_0s[:,i], m02).T, np.linalg.inv(C02)), np.subtract(self.Dtrn_0s[:,i], m02)))
            g2_scale = ((2.0*np.pi)**(-1))*(np.linalg.det(C02)**(-0.5))
            LLH += np.log((w1*g1_scale*g1_exp)+(w2*g2_scale*g2_exp))
        return LLH
    def check_convergence_for_GMM(self, i, m01, m02, C01, C02, w1, w2):
        d_m01 = np.sum(np.absolute(np.subtract(m01[i], m01[i-1])))
        d_m02 = np.sum(np.absolute(np.subtract(m02[i], m02[i-1])))
        d_C01 = np.sum(np.sum(np.absolute(np.subtract(C01[i], C01[i-1]))))
        d_C02 = np.sum(np.sum(np.absolute(np.subtract(C02[i], C02[i-1]))))
        d_w1 = abs(w1[i]-w1[i-1])
        d_w2 = abs(w2[i]-w2[i-1])
        d_params =  d_m01 + d_m02 + d_C01 + d_C02 + d_w1 + d_w2
        return (d_params<0.02)
    def part2(self, subpart):
        if subpart=='c':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D100, self.D100_Labels, self.D100_0s, self.D100_1s, self.D100_LRTs, self.D100_LRTs_0s, self.D100_LRTs_1s)
            self.Dtrn_N = 100
        elif subpart=='b':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D1k, self.D1k_Labels, self.D1k_0s, self.D1k_1s, self.D1k_LRTs, self.D1k_LRTs_0s, self.D1k_LRTs_1s)
            self.Dtrn_N = 1000
        elif subpart=='a':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D10k, self.D10k_Labels, self.D10k_0s, self.D10k_1s, self.D10k_LRTs, self.D10k_LRTs_0s, self.D10k_LRTs_1s) 
            self.Dtrn_N = 10000
        else:
            sys.exit("try again")
        self.ML_estimate_pdf()
        self.generate_gammas()
        fig_name = "ROC Curve from ML Estimated PDF ("+str(self.Dtrn_N)+" samples)"
        self.plot_ROC(self.ML_estimate, fig_name)
    class llf:
        def __init__(self, PL0, PL1, w0, w1, w2):
            self.model = "LLF"
            self.PL0 = PL0
            self.PL1 = PL1
            self.w = np.array([w0, w1, w2])
        def eval_at(self, xi):
            b = np.array([1, xi[0], xi[1]])
            exponent = (-1.0)*np.dot(self.w.T, b)
            likelihood = 1.0/(1.0+np.exp(exponent))
            return likelihood
    def ave_LLH_for_LLF(self, params):
        w0 = params[0]
        w1 = params[1]
        w2 = params[2]
        PL0 = self.Dtrn_LRTs_0s.size/self.Dtrn_N
        PL1 = self.Dtrn_LRTs_1s.size/self.Dtrn_N
        llf_h = self.llf(PL0, PL1, w0, w1, w2)
        tempsum = 0
        for n in range(self.Dtrn_N):
            h = llf_h.eval_at(self.Dtrn[:,n])
            l = self.Dtrn_Labels[n]
            tempsum += (l*np.log(h))+((1-l)*np.log(1-h))
        ave = (-1.0/self.Dtrn_N)*tempsum
        return ave
    def LLF_estimate_pdf(self):
        print("Estimating LLF from %d samples..." %self.Dtrn_N)
        init_guess = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        res = opt.minimize(self.ave_LLH_for_LLF, init_guess, method="Nelder-Mead")
        PL0 = self.Dtrn_LRTs_0s.size/self.Dtrn_N
        PL1 = self.Dtrn_LRTs_1s.size/self.Dtrn_N
        w0 = res.x[0]
        w1 = res.x[1]
        w2 = res.x[2]
        self.LLF_estimate = self.llf(PL0, PL1, w0, w1, w2)
    def part3a(self, subpart):
        if subpart=='c':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D100, self.D100_Labels, self.D100_0s, self.D100_1s, self.D100_LRTs, self.D100_LRTs_0s, self.D100_LRTs_1s)
            self.Dtrn_N = 100
        elif subpart=='b':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D1k, self.D1k_Labels, self.D1k_0s, self.D1k_1s, self.D1k_LRTs, self.D1k_LRTs_0s, self.D1k_LRTs_1s)
            self.Dtrn_N = 1000
        elif subpart=='a':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D10k, self.D10k_Labels, self.D10k_0s, self.D10k_1s, self.D10k_LRTs, self.D10k_LRTs_0s, self.D10k_LRTs_1s) 
            self.Dtrn_N = 10000
        else:
            sys.exit("try again")
        self.LLF_estimate_pdf()
        self.generate_gammas()
        fig_name = "ROC Curve from LLF Estimated PDF ("+str(self.Dtrn_N)+" samples)"
        self.plot_ROC(self.LLF_estimate, fig_name)
    class lqf:
        def __init__(self, PL0, PL1, w0, w1, w2, w3, w4, w5):
            self.model = "LQF"
            self.PL0 = PL0
            self.PL1 = PL1
            self.w = np.array([w0, w1, w2, w3, w4, w5])
        def eval_at(self, xi):
            b = np.array([1, xi[0], xi[1], xi[0]**2, xi[0]*xi[1], xi[1]**2])
            exponent = (-1.0)*np.dot(self.w.T, b)
            likelihood = 1.0/(1.0+np.exp(exponent))
            return likelihood
    def ave_LLH_for_LQF(self, params):
        w0 = params[0]
        w1 = params[1]
        w2 = params[2]
        w3 = params[3]
        w4 = params[4]
        w5 = params[5]
        PL0 = self.Dtrn_LRTs_0s.size/self.Dtrn_N
        PL1 = self.Dtrn_LRTs_1s.size/self.Dtrn_N
        lqf_h = self.lqf(PL0, PL1, w0, w1, w2, w3, w4, w5)
        tempsum = 0
        for n in range(self.Dtrn_N):
            h = lqf_h.eval_at(self.Dtrn[:,n])
            l = self.Dtrn_Labels[n]
            tempsum += (l*np.log(h))+((1-l)*np.log(1-h))
        ave = (-1.0/self.Dtrn_N)*tempsum
        return ave
    def LQF_estimate_pdf(self):
        print("Approximating LQF with LQF from %d samples..." %self.Dtrn_N)
        init_guess = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]
        res = opt.minimize(self.ave_LLH_for_LQF, init_guess, method="Nelder-Mead")
        PL0 = self.Dtrn_LRTs_0s.size/self.Dtrn_N
        PL1 = self.Dtrn_LRTs_1s.size/self.Dtrn_N
        w0 = res.x[0]
        w1 = res.x[1]
        w2 = res.x[2]
        w3 = res.x[3]
        w4 = res.x[4]
        w5 = res.x[5]
        self.LQF_estimate = self.lqf(PL0, PL1, w0, w1, w2, w3, w4, w5)
    def part3b(self, subpart):
        if subpart=='c':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D100, self.D100_Labels, self.D100_0s, self.D100_1s, self.D100_LRTs, self.D100_LRTs_0s, self.D100_LRTs_1s)
            self.Dtrn_N = 100
        elif subpart=='b':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D1k, self.D1k_Labels, self.D1k_0s, self.D1k_1s, self.D1k_LRTs, self.D1k_LRTs_0s, self.D1k_LRTs_1s)
            self.Dtrn_N = 1000
        elif subpart=='a':
            (self.Dtrn, self.Dtrn_Labels, self.Dtrn_0s, self.Dtrn_1s, self.Dtrn_LRTs, self.Dtrn_LRTs_0s, self.Dtrn_LRTs_1s) = (self.D10k, self.D10k_Labels, self.D10k_0s, self.D10k_1s, self.D10k_LRTs, self.D10k_LRTs_0s, self.D10k_LRTs_1s) 
            self.Dtrn_N = 10000
        else:
            sys.exit("try again")
        self.LQF_estimate_pdf()
        self.generate_gammas()
        fig_name = "ROC Curve from LQF Estimated PDF ("+str(self.Dtrn_N)+" samples)"
        self.plot_ROC(self.LQF_estimate, fig_name)

def main():
    main_start_time = time.time()
    ans=Q1(20000)
    ans.part1()
    ans.part2('a')
    ans.part2('b')
    ans.part2('c')
    ans.part3a('a')
    ans.part3a('b')
    ans.part3a('c')
    ans.part3b('a')
    ans.part3b('b')
    ans.part3b('c')
    main_end_time = time.time()
    total_hours, remainder = divmod(main_end_time - main_start_time, 3600)
    total_mins, total_secs = divmod(remainder, 60)
    print("Total Elapsed Time: %d:%d:%d" %(total_hours, total_mins, total_secs))

if __name__ == "__main__":
    main()

