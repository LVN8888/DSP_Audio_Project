import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path):

    signal, sr = librosa.load(file_path, sr=None)

    return signal, sr


def plot_waveform(signal):

    plt.figure(figsize=(10,4))
    plt.plot(signal)

    plt.title("Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.show()


def plot_fft(signal, sr):

    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)

    freq = np.fft.fftfreq(len(magnitude), 1/sr)

    plt.figure(figsize=(10,4))

    plt.plot(freq[:len(freq)//2], magnitude[:len(magnitude)//2])

    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.show()


def plot_spectrogram(signal, sr):

    D = librosa.stft(signal)

    S_db = librosa.amplitude_to_db(abs(D))

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Spectrogram")

    plt.show()