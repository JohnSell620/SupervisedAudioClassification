import math
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score


def get_sample_data(test):
    """
    Loads audio sample data and returns examples and labels. For each audio
    sample, the average of the left and right streams are taken.

    :param test: String identifying which test data process and return.
    :return: Examples and lables.
    :rtype: array_type
    """
    if test == 'test1':
        file_names = ['bloodycape.Wav','digitalbath.Wav','acidhologram.Wav',\
                      'stonemilker.Wav','thegate.Wav','utopia.Wav',\
                      'deuxarabesquesp1.Wav','beausoir.Wav','reverie.Wav']
        num_samples = 30
        artist = {'deftones': 1, 'bjork': 2, 'debussy': 3}
        # Labels
        Y = [artist['deftones'] if j < num_samples*3 else\
             artist['bjork'] if num_samples*3 <= j < num_samples*3*3 else\
             artist['debussy'] for j in range(num_samples*3*3)]

    elif test == 'test2':
        file_names = ['blackholesun.Wav','inbloom.Wav','alive.Wav']
        num_samples = 50
        artist = {'soundgarden': 1, 'nirvana': 2, 'pearljam': 3}
        # Labels
        Y = [artist['soundgarden'] if j < num_samples else\
             artist['nirvana'] if num_samples <= j < num_samples*2 else\
             artist['pearljam'] for j in range(num_samples*3)]

    else:
        file_names = ['sowhat.Wav','coltranesentimental.Wav','monksdream.Wav',\
                      'bolero.Wav','cinqnocturnes.Wav','princesssong.Wav',\
                      'burnthewitch.Wav','weezerhk.Wav','futurewarrior.Wav']
        num_samples = 40
        genre = {'jazz': 1, 'impressionist': 2, 'altrock': 3}
        # Labels
        Y = [genre['jazz'] if j < num_samples*3 else\
             genre['impressionist'] if num_samples*3 <=j< num_samples*3*3 else\
             genre['altrock'] for j in range(num_samples*3*3)]

    samples = []
    for j, f in enumerate(file_names):
        S, Fs = sf.read('./MusicData/' + f)
        S = (S[:,0] + S[:,1])/2 # average two signal streams
        N = len(S)

        for jj in range(num_samples):
            start_idx = int((jj+1) * N/(num_samples+1))
            samples.append(S[start_idx:start_idx+5*Fs])

    return np.asarray(samples).T, np.asarray(Y)


def svd_spectrogram(data):
    """
    Compute the SVD of the matrix constructed from flattened spectrograms of
    the audio samples.

    :param data: Matrix constructed from the audio samples.
    :return: SVD of the spectrogram matrix.
    :rtype: Numpy arrays
    """
    Fs = 44100  # Expected sample rate for all songs
    Spec_matrix = np.zeros((126936, data.shape[1])) # (129*984, X.shape[1])
    for j, S in enumerate(data.T):
        _, _, Sxx = signal.spectrogram(S, Fs, window=('gaussian', 7),
                                       return_onesided=True,
                                       mode='magnitude',
                                       scaling='spectrum')
        Spec_matrix[:,j] = Sxx.flatten()

    subsample_size = 2**2
    Spec_matrix = Spec_matrix[::subsample_size][:]
    Spec_matrix = Spec_matrix - np.mean(Spec_matrix, axis=0)
    u, s, vh = np.linalg.svd(Spec_matrix)
    return u, s, vh


def histogram_separability(data, measure):
    """
    Finds the indices of the most highly-dispersed modes by computing a
    seperability measure of the histograms.

    :param data: 2-D Numpy array containing processed signal data.
    :param measure: String indicating which measure to compute.
    :return: List of indices for the highly-dispersed modes.
    :rtype: list(int)
    """
    deviations = {}
    num_classes = 3
    xbin = np.linspace(-0.1, 0.1, 21)
    ptr = int(data.shape[0]/num_classes)
    for j in range(data.shape[1]):
        h_vecs = np.zeros((3,len(xbin)-1))
        for jj in range(num_classes):
            h, _ = np.histogram(data[jj*ptr:(jj+1)*ptr,j], bins=xbin)
            h_vecs[jj] = h

        # Compute the separability measure between all three vectors
        dm = np.zeros(3)
        reverse = True
        if measure == 'SAD':
            dm[0] = np.sum(np.abs(h_vecs[0] - h_vecs[1]))
            dm[1] = np.sum(np.abs(h_vecs[0] - h_vecs[2]))
            dm[2] = np.sum(np.abs(h_vecs[1] - h_vecs[2]))
        elif measure == 'SSD':
            dm[0] = np.sum(np.square(h_vecs[0] - h_vecs[1]))
            dm[1] = np.sum(np.square(h_vecs[0] - h_vecs[2]))
            dm[2] = np.sum(np.square(h_vecs[1] - h_vecs[2]))
        elif measure == 'PCC':
            reverse = False
            dm[0] = np.corrcoef(np.array((h_vecs[0],h_vecs[1])))[0,1]
            dm[1] = np.corrcoef(np.array((h_vecs[0],h_vecs[2])))[0,1]
            dm[2] = np.corrcoef(np.array((h_vecs[1],h_vecs[2])))[0,1]
        elif measure == 'MIS':
            dm[0] = mutual_info_score(h_vecs[0],h_vecs[1])
            dm[1] = mutual_info_score(h_vecs[0],h_vecs[2])
            dm[2] = mutual_info_score(h_vecs[1],h_vecs[2])
        deviations[j] = sum(dm)

    sorted_devs = sorted(deviations.items(), key=lambda x: x[1], reverse=reverse)
    return [k for k, v in sorted_devs if not math.isnan(v)]


def plot_model_scores(S, test, clf_names, sepmet):
    """
    Plot the predictor scores for each choice of modes retained.

    :param S: Signal data.
    :param test: String indicating which test.
    :param clf_names: Names of the classifiers.
    :param sepmet: Separarability metric used.
    """
    modes = np.linspace(2, 270, 135)
    if test == 'test2':
        modes = np.linspace(2, 150, 75)
    elif test == 'test3':
        modes = np.linspace(2, 360, 180)

    fig = plt.figure()
    for j in range(S.shape[0]):
        plt.plot(modes, S[j,:], label=clf_names[j])
    plt.gca().grid(which='major', axis='y', linestyle='--')
    plt.xlabel('Number of Modes')
    plt.ylabel('Accuracy')
    fig.suptitle('Model Accuracies\nDispersion Metric: {}'.format(sepmet))
    plt.legend(loc='lower left')
    plt.show()


def plot_mode_histogram(data, modes, test):
    """
    Plots the mode histograms.

    param: data: Array containing right singular vectors of data matrix.
    param: modes: List of modes (row indices) of data to be plotted.
    param: test: String indicating the current test.
    """
    fig = plt.figure()
    num_modes = 6
    num_classes = 3
    xbin = np.linspace(-0.1, 0.1, 21)

    ptr = int(data.shape[0]/num_classes)
    for j in range(num_modes):
        ax = fig.add_subplot(num_modes,1,j+1)

        for jj in range(num_classes):
            h, _ = np.histogram(data[jj*ptr:(jj+1)*ptr,j], bins=xbin)
            ax.plot(np.linspace(-0.1, 0.1, 20), h)
            ax.set_ylabel('Mode '+ str(modes[j]))

    fig.suptitle('Principle Modes of Test {} Data'.format(num_modes, test[4]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_sv_spectrum(s, test):
    """
    Plot the singular value spectrum, `s`.

    :param s: Singular values.
    :param test: String indicating which test.
    """
    fig = plt.figure()
    fig.suptitle('Singular Value Spectrum of X, Test {}'.format(test[4]))
    plt.plot(s,'o')
    plt.show()
