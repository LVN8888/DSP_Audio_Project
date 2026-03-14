from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt


def bandpass_filter(signal, sr):

    low = 300
    high = 8000

    b, a = butter(
        4,
        [low/(sr/2), high/(sr/2)],
        btype='band'
    )

    filtered = filtfilt(b, a, signal)

    return filtered


def plot_filter_response():

    b, a = butter(4, [300/22050, 8000/22050], btype='band')

    w, h = freqz(b, a)

    plt.figure(figsize=(8,4))

    plt.plot(w, abs(h))

    plt.title("Filter Frequency Response")
    plt.xlabel("Frequency")
    plt.ylabel("Gain")

    plt.show()