import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

class Q1:
    def __init__(self): #initialize distributions, data set, ERM model, gamma set, LDA model, tau set
        self.PL0 = 0.7
        self.PL1 = 0.3
        self.m0 = np.array([-1, -1, -1, -1])
        self.m1 = np.array([1, 1, 1, 1])
        self.C0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
        self.C1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])
        self.C0NB = np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]])
        self.C1NB = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]])
        self.N = 10000 #number of samples
        self.generate_Data()
        self.generate_gammas()
        self.estimate_mean_cov()
        self.compute_WLDA()
        self.compute_LDSs()
        self.generate_taus()
    def generate_Data(self): #draw labels and then draw data points from class-conditionals 
        tempLabels = []
        tempData = []
        tempData1 = []
        tempData0 = []
        tempERM_LRT = []
        tempNB_LRT = []
        print("Generating Data...")
        for i in range(self.N):
            if np.random.rand() > self.PL1:
                tempLabels.append(0)
                S = np.random.multivariate_normal(self.m0, self.C0)
                tempData0.append(S)
            else:
                tempLabels.append(1)
                S = np.random.multivariate_normal(self.m1, self.C1)
                tempData1.append(S)
            tempData.append(S)
            #for efficiency: LRT computations are completed here to avoid reiteration
            #would love to put LDS computations here as well, but we're married to sample averages...
            tempERM_LRT.append(self.compute_ERM_LRT(S))
            tempNB_LRT.append(self.compute_NB_LRT(S))
            i=i #vscode complains unless I use i for something...
        self.Labels = np.array(tempLabels)
        self.Data = np.transpose(np.array(tempData))
        self.Data1 = np.transpose(np.array(tempData1))
        self.Data0 = np.transpose(np.array(tempData0))
        self.ERM_LRTs = np.array(tempERM_LRT)
        self.NB_LRTs = np.array(tempNB_LRT)
    def compute_ERM_LRT(self, xi): #compute ERM LRT for a given data point
        p1_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, self.m1).T, np.linalg.inv(self.C1)), np.subtract(xi, self.m1)))
        p1_det = np.linalg.det(self.C1)**(-0.5)
        p1 = p1_det*p1_exp
        p0_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, self.m0).T, np.linalg.inv(self.C0)), np.subtract(xi, self.m0)))
        p0_det = np.linalg.det(self.C0)**(-0.5)
        p0 = p0_det*p0_exp
        return p1/p0
    def compute_NB_LRT(self, xi): #compute NB LRT for a given data point
        p1_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, self.m1).T, np.linalg.inv(self.C1NB)), np.subtract(xi, self.m1)))
        p1_det = np.linalg.det(self.C1NB)**(-0.5)
        p1 = p1_det*p1_exp
        p0_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, self.m0).T, np.linalg.inv(self.C0NB)), np.subtract(xi, self.m0)))
        p0_det = np.linalg.det(self.C0NB)**(-0.5)
        p0 = p0_det*p0_exp
        return p1/p0
    def generate_gammas(self): #create a list of gammas [0, Infinity) such that each is the midpoint between consecutive LRTs
        tempERM_gammas = []
        tempERM_LRTs = np.copy(self.ERM_LRTs)
        tempERM_LRTs.sort()
        tempNB_gammas = []
        tempNB_LRTs = np.copy(self.NB_LRTs)
        tempNB_LRTs.sort()
        for i in range(self.N):
            if i == 0:
                tempERM_gammas.append(0)
                tempNB_gammas.append(0)
            else:
                tempERM_gammas.append(tempERM_LRTs[i-1]+((tempERM_LRTs[i]-tempERM_LRTs[i-1])/2.0))
                tempNB_gammas.append(tempNB_LRTs[i-1]+((tempNB_LRTs[i]-tempNB_LRTs[i-1])/2.0))
        tempERM_gammas.append(2.0*tempERM_LRTs[self.N-1])
        self.ERM_gammas = np.array(tempERM_gammas)
        tempNB_gammas.append(2.0*tempNB_LRTs[self.N-1])
        self.NB_gammas = np.array(tempNB_gammas)
    def estimate_mean_cov(self): #estimate class-conditional means and covariances from sample averages 
        self.est_m1 = np.mean(self.Data1, axis=1, dtype=np.float64)
        self.est_m0 = np.mean(self.Data0, axis=1, dtype=np.float64)
        self.est_C1 = np.cov(self.Data1)
        self.est_C0 = np.cov(self.Data0)
    def compute_WLDA(self): #compute WLDA from estimated class-conditional means and covariances
        SB=np.array(np.outer(np.subtract(self.est_m1, self.est_m0), np.subtract(self.est_m1, self.est_m0).T))
        SW=np.array(np.add(self.est_C1, self.est_C0))
        (eigvals, eigvecs) = spla.eig(SB, SW)
        max_eigval = None
        max_eigvec = None
        for i in range(len(eigvals)):
            #print(eigvals[i])
            #print(eigvecs[i])
            if max_eigval == None:
                max_eigval = eigvals[i]
                max_eigvec = eigvecs[i]
            elif eigvals[i] > max_eigval:
                max_eigval = eigvals[i]
                max_eigvec = eigvecs[i]
        #print(max_eigval)
        #print(max_eigvec)
        self.WLDA = np.array(max_eigvec)
    def compute_LDSs(self): #compute LDSs for the entire data set
        tempLDSs = []
        for i in range(self.N):
            tempLDSs.append(np.dot(self.WLDA.T, self.Data[:,i]))
        self.LDSs = np.array(tempLDSs)
    def generate_taus(self): #create a list of taus (-Infinity, Infinity) such that each is the midpoint between consecutive LDSs
        temptaus = []
        tempLDSs = np.copy(self.LDSs)
        tempLDSs.sort()
        for i in range(self.N):
            if i == 0:
                temptaus.append(0.5*tempLDSs[i])
            else:
                temptaus.append(tempLDSs[i-1]+((tempLDSs[i]-tempLDSs[i-1])/2.0))
        temptaus.append(2.0*tempLDSs[self.N-1])
        self.taus = np.array(temptaus)
    def classify_xi_ERM(self, i, gamma): #classify a given data point by comparing its ERM LRT to gamma
        LRT = self.ERM_LRTs[i]
        if LRT > gamma:
            return 1
        elif LRT < gamma:
            return 0
        else:
            print("can't quite decide: %f >?< %f" %(LRT, gamma))
            if np.random.rand() > self.PL1:
                return 0
            else:
                return 1
    def classify_xi_NB(self, i, gamma): #classify a given data point by comparing its NB LRT to gamma
        LRT = self.NB_LRTs[i]
        if LRT > gamma:
            return 1
        elif LRT < gamma:
            return 0
        else:
            print("can't quite decide: %f >?< %f" %(LRT, gamma))
            if np.random.rand() > self.PL1:
                return 0
            else:
                return 1
    def classify_xi_LDA(self, i, tau): #classify a given data point by comparing its LDS to tau
        LDS = self.LDSs[i]
        if LDS > tau:
            return 1
        elif LDS < tau:
            return 0
        else:
            print("can't quite decide: %f >?< %f" %(LDS, tau))
            if np.random.rand() > self.PL1:
                return 0
            else:
                return 1
    def compute_confusion(self, j): #compute the confusion matrix of the data set for a given gamma/tau
        #"""
        ERM_TP = 0
        ERM_FP = 0
        ERM_TN = 0
        ERM_FN = 0
        #"""
        NB_TP = 0
        NB_FP = 0
        NB_TN = 0
        NB_FN = 0
        #"""
        LDA_TP = 0
        LDA_FP = 0
        LDA_TN = 0
        LDA_FN = 0
        #"""
        for i in range(self.N):
            #"""
            class_ERM = self.classify_xi_ERM(i, self.ERM_gammas[j])
            if (class_ERM == 1):
                if (class_ERM == self.Labels[i]):
                    ERM_TP += 1
                else:
                    ERM_FP += 1
            else:
                if (class_ERM == self.Labels[i]):
                    ERM_TN += 1
                else:
                    ERM_FN += 1
            class_NB = self.classify_xi_NB(i, self.NB_gammas[j])
            #"""
            if (class_NB == 1):
                if (class_NB == self.Labels[i]):
                    NB_TP += 1
                else:
                    NB_FP += 1
            else:
                if (class_NB == self.Labels[i]):
                    NB_TN += 1
                else:
                    NB_FN += 1
            #"""
            class_LDA = self.classify_xi_LDA(i, self.taus[j])
            if (class_LDA == 1):
                if (class_LDA == self.Labels[i]):
                    LDA_TP += 1
                else:
                    LDA_FP += 1
            else:
                if (class_LDA == self.Labels[i]):
                    LDA_TN += 1
                else:
                    LDA_FN += 1
            #"""
        #"""
        ERM_TPR = ERM_TP/(ERM_TP+ERM_FN)
        ERM_FPR = ERM_FP/(ERM_FP+ERM_TN)
        ERM_TNR = ERM_TN/(ERM_FP+ERM_TN)
        ERM_FNR = ERM_FN/(ERM_TP+ERM_FN)
        #"""
        NB_TPR = NB_TP/(NB_TP+NB_FN)
        NB_FPR = NB_FP/(NB_FP+NB_TN)
        NB_TNR = NB_TN/(NB_FP+NB_TN)
        NB_FNR = NB_FN/(NB_TP+NB_FN)
        #"""
        LDA_TPR = LDA_TP/(LDA_TP+LDA_FN)
        LDA_FPR = LDA_FP/(LDA_FP+LDA_TN)
        LDA_TNR = LDA_TN/(LDA_FP+LDA_TN)
        LDA_FNR = LDA_FN/(LDA_TP+LDA_FN)
        #"""
        return ((ERM_FPR, ERM_TPR, ERM_FNR, ERM_TNR), (NB_FPR, NB_TPR, NB_FNR, NB_TNR), (LDA_FPR, LDA_TPR, LDA_FNR, LDA_TNR))
    def plot_ROC(self): #compute the confusion matrices for all gamma/tau and then produce an ROC curve
        #"""
        ERM_FPRlist = []
        ERM_TPRlist = []
        ERM_FNRlist = []
        ERM_TNRlist = []
        ERM_min_gamma = 0
        ERM_min_P_error = 1
        ERM_min_ROC_point = [1, 1, 0, 0]
        #"""
        NB_FPRlist = []
        NB_TPRlist = []
        NB_FNRlist = []
        NB_TNRlist = []
        NB_min_gamma = 0
        NB_min_P_error = 1
        NB_min_ROC_point = [1, 1, 0, 0]
        #"""
        LDA_FPRlist = []
        LDA_TPRlist = []
        LDA_FNRlist = []
        LDA_TNRlist = []
        LDA_min_tau = 0
        LDA_min_P_error = 1
        LDA_min_ROC_point = [1, 1, 0, 0]
        #"""
        print("Beginning ERM, NB, LDA ROC Curves...")
        for i in range(self.N+1):
            ((ERM_FPR, ERM_TPR, ERM_FNR, ERM_TNR), (NB_FPR, NB_TPR, NB_FNR, NB_TNR), (LDA_FPR, LDA_TPR, LDA_FNR, LDA_TNR)) = self.compute_confusion(i)
            #"""
            ERM_FPRlist.append(ERM_FPR)
            ERM_TPRlist.append(ERM_TPR)
            ERM_FNRlist.append(ERM_FNR)
            ERM_TNRlist.append(ERM_TNR)
            ERM_P_error = ERM_FPR*self.PL0+ERM_FNR*self.PL1
            if ERM_P_error<ERM_min_P_error:
                ERM_min_P_error = ERM_P_error
                ERM_min_gamma = self.ERM_gammas[i]
                ERM_min_ROC_point = (ERM_FPR, ERM_TPR, ERM_FNR, ERM_TNR)
            #print("%d%% TPR %f FPR %f gamma %f" %(int(i/100), ERM_TPR, ERM_FPR, self.ERM_gammas[i]))
            #"""
            NB_FPRlist.append(NB_FPR)
            NB_TPRlist.append(NB_TPR)
            NB_FNRlist.append(NB_FNR)
            NB_TNRlist.append(NB_TNR)
            NB_P_error = NB_FPR*self.PL0+NB_FNR*self.PL1
            if NB_P_error<NB_min_P_error:
                NB_min_P_error = NB_P_error
                NB_min_gamma = self.NB_gammas[i]
                NB_min_ROC_point = (NB_FPR, NB_TPR, NB_FNR, NB_TNR)
            #print("%d%% TPR %f FPR %f gamma %f" %(int(i/100), NB_TPR, NB_TPR, self.NB_gammas[i]))
            #"""
            LDA_FPRlist.append(LDA_FPR)
            LDA_TPRlist.append(LDA_TPR)
            LDA_FNRlist.append(LDA_FNR)
            LDA_TNRlist.append(LDA_TNR)
            LDA_P_error = LDA_FPR*self.PL0+LDA_FNR*self.PL1
            if LDA_P_error<LDA_min_P_error:
                LDA_min_P_error = LDA_P_error
                LDA_min_tau = self.taus[i]
                LDA_min_ROC_point = (LDA_FPR, LDA_TPR, LDA_FNR, LDA_TNR)
            #print("%d%% TPR %f FPR %f tau %f" %(int(i/100), LDA_TPR, LDA_FPR, self.taus[i]))
            #"""
            if ((i%500)==0) and (i!=0):
                print("%d%%" %(int(i/100)))
        #"""
        print("ERM_gamma = %f achieves minimum P_error = %f" %(ERM_min_gamma, ERM_min_P_error))
        plt.plot(ERM_FPRlist, ERM_TPRlist, "b")
        plt.plot(ERM_min_ROC_point[0], ERM_min_ROC_point[1], "bo")
        plt.annotate("(%f, %f)\nERM_gamma = %f\nP_error = %f"%(ERM_min_ROC_point[0], ERM_min_ROC_point[1], ERM_min_gamma, ERM_min_P_error), (ERM_min_ROC_point[0], ERM_min_ROC_point[1]), (.2,.8), arrowprops={"arrowstyle": "->"})
        #"""
        print("NB_gamma = %f achieves minimum P_error = %f" %(NB_min_gamma, NB_min_P_error))
        plt.plot(NB_FPRlist, NB_TPRlist, "r")
        plt.plot(NB_min_ROC_point[0], NB_min_ROC_point[1], "ro")
        plt.annotate("(%f, %f)\nNB_gamma = %f\nP_error = %f"%(NB_min_ROC_point[0], NB_min_ROC_point[1], NB_min_gamma, NB_min_P_error), (NB_min_ROC_point[0], NB_min_ROC_point[1]), (.2,.6), arrowprops={"arrowstyle": "->"})
        #"""
        print("LDA_tau = %f achieves minimum P_error = %f" %(LDA_min_tau, LDA_min_P_error))
        plt.plot(LDA_FPRlist, LDA_TPRlist, "g")
        plt.plot(LDA_min_ROC_point[0], LDA_min_ROC_point[1], "go")
        plt.annotate("(%f, %f)\nLDA_tau = %f\nP_error = %f"%(LDA_min_ROC_point[0], LDA_min_ROC_point[1], LDA_min_tau, LDA_min_P_error), (LDA_min_ROC_point[0], LDA_min_ROC_point[1]), (.2,.4), arrowprops={"arrowstyle": "->"})
        #"""
        plt.plot([1,0],[1,0], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.suptitle("ROC Curves, True ERM vs NB ERM vs Fisher LDA")
        plt.show()

def main():
    ans=Q1()
    ans.plot_ROC()

if __name__ == "__main__":
    main()
