import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

class Q2():
    def __init__(self): #initialize distributions, data set, MAP model, ERM model, gamma set
        self.PL = [0.2, 0.25, 0.25, 0.3]
        self.m = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
        self.eigval = 0.35
        self.C = np.array([[self.eigval,0,0], [0,self.eigval,0], [0,0,self.eigval]]) #the problem statement never said each class conditional had to have a different covariance...
        self.loss_mat_MAP = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
        self.loss_mat_ERM = np.array([[0, 1, 2, 3], [10, 0, 5, 10], [20, 10, 0, 1], [30, 20, 1, 0]])
        self.N = 10000 #number of samples
        self.generate_Data()
    def generate_Data(self): #draw labels and then draw data points from class-conditionals 
        self.Labels = []
        self.Markers = []
        tempData = []
        tempPosteriors = []
        tempRisks_MAP = []
        tempRisks_ERM = []
        self.Decisions_MAP = []
        self.Decisions_ERM = []
        self.Successes_MAP = []
        self.Successes_ERM = []
        self.total_loss_MAP = 0.0
        self.total_loss_ERM = 0.0
        self.confusion_count_MAP = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.confusion_count_ERM = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        print("Generating Data...")
        for i in range(self.N):
            random_label = np.random.rand()
            if (random_label<=self.PL[0]):
                L = 1
                M = '1'
                S = np.random.multivariate_normal(self.m[0], self.C)
            elif (random_label>self.PL[0]) and (random_label<=(self.PL[0]+self.PL[1])):
                L = 2
                M = '2'
                S = np.random.multivariate_normal(self.m[1], self.C)
            elif (random_label>(self.PL[0]+self.PL[1])) and (random_label<=self.PL[0]+self.PL[1]+self.PL[2]):
                L = 3
                M = '3'
                S = np.random.multivariate_normal(self.m[2], self.C)
            elif (random_label>(self.PL[0]+self.PL[1]+self.PL[2])):
                L = 4
                M = '4'
                S = np.random.multivariate_normal(self.m[3], self.C)
            else:
                print("wtf? rand = %f" %(random_label))
            self.Labels.append(L)
            self.Markers.append(M)
            tempData.append(S)
            #for efficiency: risk computations are completed here to avoid reiteration
            post_vec = self.compute_post_vec(S)
            tempPosteriors.append(post_vec)
            risk_vec_MAP = self.compute_risk_vec_MAP(post_vec)
            risk_vec_ERM = self.compute_risk_vec_ERM(post_vec)
            tempRisks_MAP.append(risk_vec_MAP)
            tempRisks_ERM.append(risk_vec_ERM)
            D_MAP = self.decide(risk_vec_MAP)
            D_ERM = self.decide(risk_vec_ERM)
            self.Decisions_MAP.append(D_MAP)
            self.Decisions_ERM.append(D_ERM)
            if D_MAP == L:
                self.Successes_MAP.append('success')
            else:
                self.Successes_MAP.append('failure')
            if D_ERM == L:
                self.Successes_ERM.append('success')
            else:
                self.Successes_ERM.append('failure')
            self.total_loss_MAP += self.loss_mat_MAP[D_MAP-1, L-1]
            self.total_loss_ERM += self.loss_mat_ERM[D_ERM-1, L-1]
            self.confusion_count_MAP[D_MAP-1, L-1] += 1
            self.confusion_count_ERM[D_ERM-1, L-1] += 1
            i=i #vscode complains unless I use i for something...
        self.Data = np.transpose(np.array(tempData))
        self.Posteriors = np.transpose(np.array(tempPosteriors))
        self.Risks_MAP = np.transpose(np.array(tempRisks_MAP))
        self.Risks_ERM = np.transpose(np.array(tempRisks_ERM))
        self.expected_loss_MAP = self.total_loss_MAP/self.N
        self.expected_loss_ERM = self.total_loss_ERM/self.N
        self.compute_confusion_MAP()
        self.compute_confusion_ERM()
    def compute_post_vec(self, xi):
        post_vec = []
        for i in range(4):
            cc_det = (np.linalg.det(self.C))**(-0.5)
            cc_exp = np.exp((-0.5)*np.dot(np.dot(np.subtract(xi, self.m[i]).T, np.linalg.inv(self.C)), np.subtract(xi, self.m[i])))
            cc = cc_det*cc_exp
            post_vec.append(cc*self.PL[i])
        return post_vec
    def compute_risk_vec_MAP(self, post_vec):
        return np.dot(self.loss_mat_MAP, np.array(post_vec).T)
    def compute_risk_vec_ERM(self, post_vec):
        return np.dot(self.loss_mat_ERM, np.array(post_vec).T)
    def decide(self, risk_vec):
        return (1+np.argmin(risk_vec))
    def compute_confusion_MAP(self):
        self.confusion_MAP = np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
        for i in range(4):
            for j in range(4):
                self.confusion_MAP[i,j] = self.confusion_count_MAP[i,j]/((self.confusion_count_MAP[:,j]).sum())
    def compute_confusion_ERM(self):
        self.confusion_ERM = np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
        for i in range(4):
            for j in range(4):
                self.confusion_ERM[i,j] = self.confusion_count_ERM[i,j]/((self.confusion_count_ERM[:,j]).sum())
    def plot_MAP(self):
        #prints
        print("Printing MAP results...")
        for i in range(4):
            print("Success rate for Label %d: %f" %(i+1, self.confusion_MAP[i,i]))
        print("Full confusion matrix:")
        print(self.confusion_MAP)
        print("Expected loss: %f" %self.expected_loss_MAP)
        #plots
        table = {'x': self.Data[0,:], 'y': self.Data[1,:], 'z': self.Data[2,:], 'Markers': self.Markers, 'suc_MAP': self.Successes_MAP}
        palette = {'success': 'g', 'failure': 'r'}
        markers = {'1': '.', '2': 'o', '3': '^', '4': 's'}
        df = pd.DataFrame(data=table)
        sb.scatterplot(data=df, x='x', y='y', hue='suc_MAP', style='Markers', palette=palette, markers=markers)
        plt.suptitle("MAP Results")
        plt.show()
    def plot_ERM(self):
        #prints
        print("Printing ERM results...")
        for i in range(4):
            print("Success rate for Label %d: %f" %(i+1, self.confusion_ERM[i,i]))
        print("Full confusion matrix:")
        print(self.confusion_ERM)
        print("Expected loss: %f" %self.expected_loss_ERM)
        #plots
        table = {'x': self.Data[0,:], 'y': self.Data[1,:], 'z': self.Data[2,:], 'Markers': self.Markers, 'suc_ERM': self.Successes_ERM}
        palette = {'success': 'g', 'failure': 'r'}
        markers = {'1': '.', '2': 'o', '3': '^', '4': 's'}
        df = pd.DataFrame(data=table)
        sb.scatterplot(data=df, x='x', y='y', hue='suc_ERM', style='Markers', palette=palette, markers=markers)
        plt.suptitle("ERM Results")
        plt.show()
    def plot_results(self):
        self.plot_MAP()
        self.plot_ERM()
        print("Change in Confusion[i,i] from MAP to ERM...")
        for i in range(4):
            print("Change in success rate for Label %d: %f" %(i+1, self.confusion_MAP[i,i]-self.confusion_ERM[i,i]))


def main():
    ans=Q2()
    ans.plot_results()

if __name__ == "__main__":
    main()

