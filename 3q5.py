import numpy as np
import time
import matplotlib.pyplot as plt
n_values = range(4, 101)  # n from 4 to 100
# Number of trials for averaging
num_trials = 10
# Arrays to store average computation times
time_dft_manual_avg = []
time_dft_numpy_avg = []
for n in n_values:
    data = np.arange(n, dtype=np.complex_)
 # Perform manual DFT computation trials
    manual_times = []
    for _ in range(num_trials):
        start_time = time.time()
        dft_data = np.zeros(n, dtype=np.complex_)
        for i in range(n):
            for m in range(n):
                dft_data[i] += data[m] * np.exp((-1j * m * i * 2 * np.pi) / n)
            dft_data[i] /= np.sqrt(n)
        manual_times.append(time.time() - start_time)
    time_dft_manual_avg.append(np.mean(manual_times))
# Perform numpy.fft.fft computation trials
    numpy_times = []
    for _ in range(num_trials):
        start_time = time.time()
        np_dft_data = np.fft.fft(data, norm='ortho')
        numpy_times.append(time.time() - start_time)
    time_dft_numpy_avg.append(np.mean(numpy_times))
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_values, time_dft_manual_avg, "-", label='Manual Computation (Average)')
plt.plot(n_values, time_dft_numpy_avg, "-", label='Computation using numpy.fft.fft (Average)')
plt.title('Average Computation Time vs. n')
plt.xlabel('n (Length of Input)')
plt.ylabel('Average Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
