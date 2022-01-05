import numpy as np
import matplotlib.pyplot as plt
import os.path as path

#just a quick and dirty plotting script
def main():
    num_gaussians = 10 #req 10+
    data_power = [2,3,4,5] #req [2,3,4,5] - 6 if I have time
    num_experiments = 100 #req 100+
    BIC_results = []
    BIC_labels = []
    Kfold_results = []
    Kfold_labels = []
    for p in range(len(data_power)):
        BIC_file_path = "BIC_results_for_p="+str(data_power[p])+".npy"
        if path.exists(BIC_file_path):
            with open(BIC_file_path, "rb") as fp:
                BIC_results.append(np.load(fp))
                BIC_labels.append("p="+str(data_power[p]))
        Kfold_file_path = "Kfold_results_for_p="+str(data_power[p])+".npy"
        if path.exists(Kfold_file_path):
            with open(Kfold_file_path, "rb") as fp:
                Kfold_results.append(np.load(fp))
                Kfold_labels.append("p="+str(data_power[p]))
    plt.clf()
    plt.suptitle("Box&Whisker plots of BIC for GMM model order selection")
    plt.xlabel("Training with N=10^p samples")
    plt.ylabel("Selection of model order")
    plt.boxplot(BIC_results, labels=BIC_labels)
    plt.savefig("BIC_model_order_results") #plt.show()
    plt.clf()
    plt.suptitle("Box&Whisker plots of Kfold for GMM model order selection")
    plt.xlabel("Training with N=10^p samples")
    plt.ylabel("Selection of model order")
    plt.boxplot(Kfold_results, labels=Kfold_labels)
    plt.savefig("Kfold_model_order_results") #plt.show()

if __name__ == "__main__":
    main()

