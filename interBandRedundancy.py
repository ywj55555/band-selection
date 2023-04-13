import utilsF as utils
import statsmodels.api as sm
import numpy as np
import torch
import os
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from TrainSelection import TrainSelection

def mkdir(path):
    path = path.strip()
    path = path.rstrip('\\')
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path,exist_ok=True)
        print("create dir(" + path + ") successfully")
        return True
    else:
        print("dir(" + path + ") is exist")
        return False

class InterBandRedundancy:

    def __init__(self, flag_average=True, median=True, normalize=True, printProcess=True, dataset="Kochia",
                 threshold=10):
        """Constructor. LOAD HDF5 FILE"""
        self.train_x, self.train_y, ind = utils.load_data(flag_average=flag_average, normalization=normalize,
                                                          median=median, data=dataset)
        self.printProcess = printProcess
        self.threshold = threshold
        self.table = np.zeros((self.train_x.shape[3], self.train_x.shape[3]))

    def setT(self, threshold):
        """Set VIF threshold"""
        self.threshold = threshold

    def vifPair(self, i, i2):
        """Calculates VIF value between i-th and i2-th bands"""
        y = self.train_x[:, :, :, i]  # Sets dependant variable
        x = self.train_x[:, :, :, i2]  # Sets independent variable
        # Reshape images into a 1-D vector
        x = x.reshape((self.train_x.shape[0] * self.train_x.shape[1] * self.train_x.shape[2], 1))
        y = y.reshape((self.train_x.shape[0] * self.train_x.shape[1] * self.train_x.shape[2], 1))
        model = sm.OLS(y, x)  # OLS regression
        results = model.fit()
        rsq = results.rsquared  # Gets R^2 value
        VIFValue = round(1 / (1 - rsq), 2)  # Computes VIF value
        if self.printProcess:
            print("Comparing band: " + str(i) + " and band: " + str(i2) + ". VIF: " + str(VIFValue))
        return VIFValue.astype(np.float32)

    def clusters(self):

        distances_left = np.zeros((self.train_x.shape[3]))
        distances_right = np.zeros((self.train_x.shape[3]))

        for band in range(0, self.train_x.shape[3]):
            # Check left
            d = 1  # Set initial distance
            vifVal = np.infty
            while vifVal > self.threshold and (band - d) > 0:
                print("Evaluating band ", band, " with a distance ", d)
                if self.table[band, band - d] == 0:
                    self.table[band, band - d] = self.vifPair(band, band - d)
                    self.table[band - d, band] = self.table[band, band - d]
                vifVal = self.table[band, band - d]
                d += 1
            distances_left[band] = d - 1

            # Check right
            d = 1  # Set initial distance
            vifVal = np.infty
            while vifVal > self.threshold and (band + d) < self.train_x.shape[3]:
                print("Evaluating band ", band, " with a distance ", d)
                if self.table[band, band + d] == 0:
                    self.table[band, band + d] = self.vifPair(band, band + d)
                    self.table[band + d, band] = self.table[band, band + d]
                vifVal = self.table[band, band + d]
                d += 1
            distances_right[band] = d - 1

        return list(np.abs(distances_left - distances_right))

    def plotSample(self):
        """Plot the reflectance response and the distance plot"""
        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 0)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2), (self.train_x.shape[3],
                                                                 self.train_x.shape[0] * self.train_x.shape[1] *
                                                                 self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means0 = np.mean(sample, axis=1)
        stds0 = np.std(sample, axis=1)

        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 1)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2),
                            (self.train_x.shape[3], self.train_x.shape[0] * self.train_x.shape[1] *
                             self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means1 = np.mean(sample, axis=1)
        stds1 = np.std(sample, axis=1)

        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 2)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2),
                            (self.train_x.shape[3], self.train_x.shape[0] * self.train_x.shape[1] *
                             self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means2 = np.mean(sample, axis=1)
        stds2 = np.std(sample, axis=1)

        # Plot graph
        fig, ax = plt.subplots()
        ax.set_ylabel('Reflectance')
        ax.set_xlabel('n : Band Index')
        clrs = sns.color_palette()
        with sns.axes_style("darkgrid"):
            epochss = list(range(self.train_x.shape[3]))
            ax.plot(epochss, means0, c=clrs[0], linestyle='-', linewidth=2.5, label='Susceptible Kochia')
            ax.fill_between(epochss, means0 - stds0, means0 + stds0, alpha=0.3, facecolor=clrs[0])
            ax.legend()
            ax.plot(epochss, means1, c=clrs[1], linestyle=':', linewidth=2.5, label='Glyphosate Resistant')
            ax.fill_between(epochss, means1 - stds1, means1 + stds1, alpha=0.3, facecolor=clrs[1])
            ax.legend()
            ax.plot(epochss, means2, c=clrs[2], linestyle='-.', linewidth=3, label='Dicamba Resistant')
            ax.fill_between(epochss, means2 - stds2, means2 + stds2, alpha=0.3, facecolor=clrs[2])
            ax.legend()
        mkdir('./Kochia//plots')
        with open("./Kochia//plots//Kochia_distances_VIF12", 'rb') as fil:
            ds = list(pickle.load(fil))
        ax2 = ax.twinx()  # position of the xticklabels in the old x-axis
        ax2.set_ylabel('d(x_n) = |d_left(x_n) - d_right(x_n)|')
        ax2.plot(epochss, ds, c='k')
        ds.insert(0, 100)  # Add high values at the beginning and the end so that initial
        ds.append(100)
        indx, _ = find_peaks(np.max(ds) - ds, height=0)
        ds = np.array(ds[1:-1])
        indx = indx - 1
        # Remove points with a distance greater or equal than 5 (not suitable centers)
        indx = [p for p in indx if ds[p] < 5]
        ax2.plot(np.array(indx), ds[np.array(indx)], "x", c='r')
        ax3 = ax.twiny()
        newlabel = list(range(387, 1023 + 50, 50))
        funct = lambda tx: (tx - 387) / 4.24 + 10
        newpos = [funct(tx) for tx in newlabel]
        ax3.set_xlim([0, 150])
        ax3.set_xticks(newpos)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_xticklabels(newlabel)
        plt.savefig('VIF12plusReflectance.png', dpi=600)


if __name__ == '__main__':

    data = 'IP'  # Specify the dataset to be analyzed
    # data = 'Skin'  # Specify the dataset to be analyzed
    classifier = 'CNN'  #
    # nbands = 5  # Specify the number of desired bands
    nbands = 9  # Specify the number of desired bands
    size = 100
    average = True
    medianF = False
    batch = 128
    if data == 'IP' or data == 'PU':
        average = False
    elif data == 'Avocado':
        medianF = True
        batch = 8

    sizestr = ''
    if size != 100:
        sizestr = str(size)

    epochs = 130
    if classifier == "ANN":
        epochs = 90
        batch = 2048  # 1024 for Kochia

    interB = InterBandRedundancy(dataset=data, flag_average=average, normalize=True)
    # interB.plotSample()
    th = 12  # VIF threshold

    for t in reversed(range(5, th + 1)):  # Test values from 10 to 5
        print("VIF THRESHOLD: " + str(t))

        # Check if the analysis have been made before
        filepreselected = './' + data + "//results//SSA//preselection_" + data + "_VIF" + str(t)
        filedistances = './' + data + "//results//SSA//distances_" + data + "_VIF" + str(t)
        fileselected = './' + data + "//results//SSA//" + str(nbands) + " bands//selection_" + data + sizestr + classifier + \
                       str(nbands) + "bands_VIF" + str(t) + ".txt"
        # if not os.path.exists('./' + data + "//results//SSA//"):
        #     mkdir('./' + data + "//results//SSA//")
        if os.path.exists(filepreselected):
            with open(filepreselected, 'rb') as f:
                indexes = pickle.load(f)
        else:
            interB.setT(t)
            # Get the distribution of distances
            dist = interB.clusters()
            # Calculate local minima
            dist.insert(0, 100)  # Add high values at the beginning and the end so that initial
            dist.append(100)  # and final bands can be considered as local minima
            indexes, _ = find_peaks(np.max(dist) - dist, height=0)
            dist = np.array(dist[1:-1])  # Remove the dummy points previously added
            indexes = indexes - 1
            # Remove points with a distance greater or equal than 5 (not suitable centers)
            indexes = [p for p in indexes if dist[p] < 5]

            # Save pre-selected bands for VIF value of t
            dir_path_tmp, file_name = os.path.split(filepreselected)
            if not os.path.exists(dir_path_tmp):
                mkdir(dir_path_tmp)
            with open(filepreselected, 'wb') as fi:
                pickle.dump(indexes, fi)
            # Save distribution of distances for VIF value of t
            dir_path_tmp, file_name = os.path.split(filedistances)
            if not os.path.exists(dir_path_tmp):
                mkdir(dir_path_tmp)
            with open(filedistances, 'wb') as fi:
                pickle.dump(dist, fi)
        print('preselected', indexes)
        # Get the k-selected bands based on IE
        net = TrainSelection(method='SSA', classifier=classifier, transform=False, average=average, batch_size=batch,
                             epochs=epochs, plot=False, selection=indexes, th=str(t), data=data, median=medianF,
                             size=size)
        index, entr = net.selection(select=nbands)
        # Save selected bands as txt file
        print('final selected', index)
        dir_path_tmp, file_name = os.path.split(fileselected)
        if not os.path.exists(dir_path_tmp):
            mkdir(dir_path_tmp)
        with open(fileselected, 'w') as x_file:
            x_file.write(str(index))
        # Save scores of each of the bands
        tmp_path = './' + data + "//results//SSA//bandScores_" + data + sizestr + classifier + "_VIF" + str(t)
        dir_path_tmp, file_name = os.path.split(tmp_path)
        if not os.path.exists(dir_path_tmp):
            mkdir(dir_path_tmp)
        with open(tmp_path, 'wb') as fi:
            pickle.dump(entr, fi)

        # Train selected bands if the selected set of bands was not trained before
        if not os.path.exists(data + "//results//SSA//" + str(nbands) + " bands//classification_report5x2_" + sizestr +
                              classifier + "SSA" + str(nbands) + data + str(t) + ".txt"):
            np.random.seed(seed=7)  # Re-Initialize seed to get reproducible results
            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            net = TrainSelection(method='SSA', classifier=classifier, transform=False, average=average,
                                 batch_size=batch, epochs=epochs, median=medianF, plot=False, selection=index,
                                 th=str(t), data=data, size=size)
            net.train()
            # net.validate()  # Store the evaluation metrics