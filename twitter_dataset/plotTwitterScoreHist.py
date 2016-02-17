import numpy as np
import matplotlib.pyplot as plt
import sys

histogram_bins = 20

if __name__ == '__main__':
    path = "/imatge/vcampos/work/twitter_dataset/scores/"

    if (len(sys.argv) >= 2):
        try:
            path = path + str(sys.argv[1]) + '/'
        except:
            sys.exit('Wrong parameters. The architecture is needed.')
    else:
        sys.exit('Not enough parameters')



    # Load the data
    positiveScores = np.load(path+"positiveScores.npy")
    negativeScores = np.load(path+"negativeScores.npy")

    # Compute histograms
    nx, xbins, ptchs = plt.hist(positiveScores, bins=histogram_bins, normed=1, histtype='stepfilled')
    plt.clf() # Get rid of this histogram since not the one we want.
    nx2, xbins2, ptchs2 = plt.hist(negativeScores, bins=histogram_bins, normed=1, histtype='stepfilled')
    plt.clf() # Get rid of this histogram since not the one we want.

    # Force the histograms tu sum 1
    nx_frac = nx/float(len(nx)) # Each bin divided by total number of objects.
    width = xbins[1] - xbins[0] # Width of each bin.
    x = np.ravel(zip(xbins[:-1], xbins[:-1]+width))
    y = np.ravel(zip(nx_frac,nx_frac))
    nx_frac2 = nx2/float(len(nx2)) # Each bin divided by total number of objects.
    width2 = xbins2[1] - xbins2[0] # Width of each bin.
    x2 = np.ravel(zip(xbins2[:-1], xbins2[:-1]+width2))
    y2 = np.ravel(zip(nx_frac2,nx_frac2))

    # Move the last bin to 1
    xbins[-2]=xbins[-2]+width
    xbins2[-2]=xbins2[-2]+width2

    # Create figure and plot
    fig = plt.figure(1)
    plt.plot(xbins[:-1],nx_frac,linestyle="solid", color="g", label="Positive sentiment")
    plt.plot(xbins2[:-1],nx_frac2,linestyle="solid", color="r", label="Negative sentiment")

    plt.title("Scores hist")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
