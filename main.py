import numpy as np
import random
import matplotlib.pyplot as plt
import time as t

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

def interval(start, stop, step):
    signals = []
    N = []

    for i in range(start, stop + 1, step):
        signals.append(gen_signal(n, W_max, i))
        N.append(i)

    return [N, signals] 

def get_time(interval, ft):
    time = []
    N = interval[0]
    signal = interval[1]

    for i in range(0, len(N)):
        start_time = t.time()
        ft(signal[i])
        time.append(t.time() - start_time)

    return [N, time]
    
    
n = int(input("Harmonics = "))
W_max = int(input("Frequency = "))
N = int(input("Number of points = "))
                            
signal = gen_signal(n, W_max, N)

spectrum_dft = dft(signal)
spectrum_fft = fft(signal)
spectrum_numpy_fft = np.fft.fft(signal)

signal_interval = interval(100, 1500, 100)
time_dft = get_time(signal_interval, dft)
time_fft = get_time(signal_interval, fft)


plt.figure(1, figsize=(10, 7))

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


plt.figure(2, figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.title('Time of dft')
plt.plot(time_dft[0], time_dft[1], 'k')

plt.subplot(3, 1, 2)
plt.title('Time of fft')
plt.plot(time_fft[0], time_fft[1], 'k')

plt.subplot(3, 1, 3)
plt.title('Comparison of dft and fft')
plt.plot(time_dft[0], time_dft[1], label = "dft")
plt.plot(time_fft[0], time_fft[1], label = "fft")
plt.legend()

plt.tight_layout()
plt.show()