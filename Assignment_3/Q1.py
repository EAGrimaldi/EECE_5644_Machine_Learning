import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as pp
import os.path

class Q1_GMM:
    def __init__(self):
        #initializes a very simple GMM with 4 components on the corners of a square and about 10-20% overlap
        self.C = 4
        self.alpha = [0.25, 0.25, 0.25, 0.25]
        '''
        #First Distribution (very messy)
            mu0 = np.array([1.0,1.0,0.0])
            mu1 = np.array([-1.0,-1.0,0.0])
            mu2 = np.array([-1.0,0.0,0.5])
            mu3 = np.array([0.5,-1.0,0.0])
            self.mu = [mu0, mu1, mu2, mu3]
            sig0 = np.array([\
                [0.1,0.0,0.0],\
                [0.0,0.5,0.3],\
                [0.0,0.3,0.4]])
            sig1 = np.array([\
                [0.3,0.0,0.0],\
                [0.0,0.5,0.0],\
                [0.0,0.0,0.3]])
            sig2 = np.array([\
                [0.4,0.0,0.0],\
                [0.0,0.4,0.0],\
                [0.0,0.0,0.4]])
            sig3 = np.array([\
                [0.2,0.0,0.0],\
                [0.0,0.2,0.0],\
                [0.0,0.0,0.2]])
            self.sigma = [sig0, sig1, sig2, sig3]
        '''
        #Second Distribution (very clean)
        mu0 = np.array([1.0,1.0,0.0])
        mu1 = np.array([1.0,-1.0,0.0])
        mu2 = np.array([-1.0,-1.0,0.0])
        mu3 = np.array([-1.0,1.0,0.0])
        self.mu = [mu0, mu1, mu2, mu3]
        var = 0.5
        sig = np.array([\
            [var,0.0,0.0],\
            [0.0,var,0.0],\
            [0.0,0.0,var]])
        self.sigma = [sig, sig, sig, sig]
    class dataset:
        def __init__(self, stuff):
            self.N = stuff[0]
            self.K = stuff[1]
            self.D = stuff[2]
            self.L = stuff[3]
            self.D_K = stuff[4]
            self.L_K = stuff[5]
            self.D_not_K = stuff[6]
            self.L_not_K = stuff[7]
    def draw_samples(self, N, K=10):
        tempData = []
        tempLabels = []
        tempDataByKfold = []
        tempLabelsByKfold = []
        tempDataByNotKfold = []
        tempLabelsByNotKfold = []
        for k in range(K):
            k=k #quiet pylint
            tempDataByKfold.append([])
            tempLabelsByKfold.append([])
            tempDataByNotKfold.append([])
            tempLabelsByNotKfold.append([])
        k=0
        for i in range(N):
            i=i #quiet pylint
            rand_comp = np.random.randint(0,high=self.C)
            sample = np.random.multivariate_normal(self.mu[rand_comp], self.sigma[rand_comp])
            tempData.append(sample)
            one_hot = []
            for c in range(self.C):
                if c==rand_comp:
                    one_hot.append(1)
                else:
                    one_hot.append(0)
            tempLabels.append(one_hot)
            tempDataByKfold[k].append(sample)
            tempLabelsByKfold[k].append(one_hot)
            for j in range(K):
                if j!=k:
                    tempDataByNotKfold[j].append(sample)
                    tempLabelsByNotKfold[j].append(one_hot)
            k += 1
            if (k >= K):
                k = 0
        Data = np.array(tempData)
        Labels = np.array(tempLabels)
        Data_by_Kfold = np.array(tempDataByKfold)
        Labels_by_Kfold = np.array(tempLabelsByKfold)
        Data_by_not_Kfold = np.array(tempDataByNotKfold)
        Labels_by_not_Kfold = np.array(tempLabelsByNotKfold)
        return self.dataset([N, K, Data, Labels, Data_by_Kfold, Labels_by_Kfold, Data_by_not_Kfold, Labels_by_not_Kfold])
    def classify_sample(self, x):
        risk = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
        temp = []
        for i in range(self.C):
            scale = 1.0/(np.sqrt(np.linalg.det(self.sigma[i])))
            exponent = (-0.5)*np.dot(np.dot(np.subtract(x, self.mu[i]).T, np.linalg.inv(self.sigma[i])), np.subtract(x, self.mu[i]))
            temp.append(scale*np.exp(exponent))
        post = np.array(temp)
        full = np.dot(risk, post)
        guess = np.argmin(full)
        return guess
    def classify_dataset(self, dataset):
        success = 0
        failure = 0
        for i in range(dataset.N):
            guess = self.classify_sample(dataset.D[i,:])
            truth = np.argmax(dataset.L[i,:])
            if guess == truth:
                success += 1
            else:
                failure += 1
        print("Empirical results of theoretically optimal classifier")
        print("Success rate: %f" %(float(success)/float(dataset.N)))
        print("Failure rate: %f" %(float(failure)/float(dataset.N)))
        return float(failure)/float(dataset.N)
    def init_data(self, N, K=10, save=True):
        folder = 'D'+str(N)+'\\'
        parts = ['D','L','D_K','L_K','D_not_K','L_not_K']
        if os.path.exists(folder):
            stuff = [N, K]
            for part in parts:
                file_path = folder+part+'.npy'
                with open(file_path, 'rb') as fp:
                    stuff.append(np.load(fp))
            temp = self.dataset(stuff)
        else:
            temp = self.draw_samples(N, K)
            stuff = [temp.D, temp.L, temp.D_K, temp.L_K, temp.D_not_K, temp.L_not_K]
            if save:
                os.makedirs(folder)
                for i in range(6):
                    file_path = folder+parts[i]+'.npy'
                    with open(file_path, 'wb') as fp:
                        np.save(fp, stuff[i])
        return temp

def get_model(p, c=4):
    model = ks.Sequential([\
        ks.layers.Dense(p, activation='elu'),\
        ks.layers.Dense(c, activation='softmax')])
    model.compile(optimizer='adam', loss=ks.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def model_data(name, Dtrain, Dtest, save=True):
    #use saved model if possible
    folder = name+'\\final_model'
    if os.path.exists(folder):
        file_path = folder+'\\perceptron_choice.npy'
        with open(file_path, 'rb') as fp:
            temp = np.load(fp, allow_pickle=True)
            perceptron_choice = temp[0]
            loss = temp[1]
            acc = temp[2]
        model = ks.models.load_model(folder)
        print("For %d samples, best number of perceptrons is %d, minimum probability of error is %f."%(Dtrain.N, perceptron_choice, (1-acc)))
        return (1-acc)
    else:
        #choose number of perceptrons with KFCV
        max_perceptrons = 50 #50 honestly seems a little unreasonable given the simplicity of the data
        K = 10
        average = []
        del_average = []
        converged = False
        p=0
        while not converged and p<max_perceptrons:
            print("Trying %d perceptrons..." %(p+1))
            tempsum = 0
            for k in range(K):
                Dtrn = Dtrain.D_not_K[k]
                Ltrn = Dtrain.L_not_K[k]
                Dval = Dtrain.D_K[k]
                Lval = Dtrain.L_K[k]
                model = get_model(p+1)
                model.fit(Dtrn, Ltrn, epochs=10, verbose=0)
                #print("Evaluation")
                loss, acc = model.evaluate(Dval, Lval, verbose=0)
                tempsum += 1-acc
            average.append(tempsum/K)
            #due to quirks of the random data, stochastic gradient descent, and random initialization, KFCV found a floor instead of a valley
            #in order to prevent excessive perceptrons, we decide KFCV has converged if the last 10 experiments result in a change of less than 1% error
            if p==0:
                del_average.append(average[p])
            elif p>0:
                del_average.append(average[p]-average[p-1])
            if p>9:
                small_changes = True
                for q in range(10):
                    if abs(del_average[p-q])>0.01:
                        small_changes = False
                if small_changes:
                    converged = True
            p +=1
        if converged:
            print('converged')
            perceptron_choice = p-10
        else:
            perceptron_choice = np.argmin(average)+1
        pp.plot(np.linspace(0,p-1,num=p),average)
        pp.savefig('D%d_error_vs_nodes' %Dtrain.N)
        #choose best model out of 10 random initializations
        rand_init_loss = []
        for r in range(10):
            model = get_model(perceptron_choice)
            model.fit(Dtrain.D, Dtrain.L, epochs=10, verbose=0)
            loss, acc = model.evaluate(Dtrain.D, Dtrain.L, verbose=0)
            rand_init_loss.append(loss)
            folder = name+'\\rand_init_'+str(r)
            os.makedirs(folder)
            model.save(folder)
        rand_init_choice = np.argmin(rand_init_loss)
        folder = name+'\\rand_init_'+str(rand_init_choice)
        #classify test set and empirically estimate minimum probability of error
        model = ks.models.load_model(folder)
        loss, acc = model.evaluate(Dtest.D, Dtest.L, verbose=1)
        print("For %d samples, best number of perceptrons is %d, minimum probability of error is %f."%(Dtrain.N, perceptron_choice, (1-acc)))
        #save model for future use
        if save:
            folder = name+'\\final_model'
            os.makedirs(folder)
            model.save(folder)
            file_path = folder+'\\perceptron_choice.npy'
            with open(file_path, 'wb') as fp:
                np.save(fp, np.array([perceptron_choice, loss, acc]))
        return (1-acc)

def main():
    #load datasets
    PDF = Q1_GMM()
    D1h = PDF.init_data(100)
    D2h = PDF.init_data(200)
    D5h = PDF.init_data(500)
    D1k = PDF.init_data(1000)
    D2k = PDF.init_data(2000)
    D5k = PDF.init_data(5000)
    Dtest = PDF.init_data(100000)
    #empirical estimate of probability error for theoretically optimal classifier
    theoretical_minPe = PDF.classify_dataset(Dtest)
    #train a model for each dataset
    name = ["D100","D200","D500","D1000","D2000","D5000"]
    data = [D1h,D2h,D5h,D1k,D2k,D5k]
    num_samples = [100,200,500,1000,2000,5000]
    empirical_minPe = []
    for i in range(6):
        #print("Trying %s..." %(name[i]))
        minPe = model_data(name[i], data[i], Dtest)
        empirical_minPe.append(minPe)
    pp.clf
    pp.scatter(num_samples, empirical_minPe)
    pp.plot(num_samples, empirical_minPe)
    pp.axhline(theoretical_minPe, linestyle="--")
    pp.show()

if __name__ == "__main__":
    main()
