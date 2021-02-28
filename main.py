import numpy as np
import random
import matplotlib.pyplot as plt

def gen_signal(n, W_max, N):
    signals = [0] * N
    step = W_max/n
    
    for i in range(n):
        A = random.random()
        fi = random.random()
        w = i * step
        for t in range(N):
            signals[t] += A * np.sin(w * t + fi)
    return signals

def dft(signal):
    N = len(signal)
    spectrum = [0] * N

    for n in range(0, N):
        for k in range(0, N):
            spectrum[n] += signal[k] * np.exp(-2j * np.pi * k * n / N)
    return spectrum


def fft(signal):
    N = len(signal)
    spectrum = [0] * N

    if N == 1: return signal

    left_part = [signal for i, signal in enumerate(signal) if i % 2 == 0]
    right_part = [signal for i, signal in enumerate(signal) if i % 2 == 1]

    transformed_left = fft(left_part)
    transformed_right = fft(right_part)

    inverted_root = np.exp(-2j * np.pi / N)
    root = 1

    for i in range(0, int(N/2)):
        spectrum[i] = transformed_left[i] + root * transformed_right[i]
        spectrum[int(i + N/2)] = transformed_left[i] - root * transformed_right[i]
        root = root * inverted_root
    return spectrum


n = int(input("Harmonics = "))
W_max = int(input("Frequency = "))
N = int(input("Number of points = "))
                            
signal = gen_signal(n, W_max, N)

spectrum_dft = dft(signal)
spectrum_fft = fft(signal)
spectrum_numpy_fft = np.fft.fft(signal)

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.title('Signal')
plt.plot(signal, 'k')

plt.subplot(2, 2, 2)
plt.title('Spectrum(dft)')
plt.plot(np.abs(spectrum_dft), 'k')

plt.subplot(2, 2, 3)
plt.title('Spectrum(fft)')
plt.plot(np.abs(spectrum_fft), 'k')

plt.subplot(2, 2, 4)
plt.title('Spectrum(numpy.fft)')
plt.plot(np.abs(spectrum_numpy_fft), 'k')

plt.tight_layout()
plt.show()