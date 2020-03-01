""" @author: Muhammad Ali Qureshi """

# Importing libraries
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Extracting and Sampling Voice Data
samplerate, data = wavfile.read('C:/...wave2.wav')
samplerate1, data1 = wavfile.read('C:/.../wave1.wav')
samplerate2, data2 = wavfile.read('C:/...wave.wav')
times = np.arange(len(data))/float(samplerate)
times1 = np.arange(len(data1))/float(samplerate1)
times2 = np.arange(len(data2))/float(samplerate2)

# Slicing Voice Data
data0=data[:,0];data00=data1[:,0];data000=data2[:,0]
cut1=data0;cut2=data00;cut3=data000

# Lorenz Attractor for Generating Chaos
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

dt = 0.01
num_steps = len(cut2)-1

# Need One More For The Initial Values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set Initial Values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

xr=xs;yr=ys;zr=zs
    
#Select Time and Wave
time=times1; wave=cut2

xr=np.int16(xr); yr=np.int16(yr); zr=np.int16(zr)
xr=xr*10000; yr=yr*10000; zr=zr*10000

"""Encryption Portion"""
x=np.bitwise_xor(xr,yr)
x1=np.bitwise_xor(x,zr)
encr1=np.bitwise_xor(x1,wave) 

"""Decryption Portion"""
decr1=np.bitwise_xor(encr1,x1)
encr=encr1
decr=decr1

"""Ploting Portion for Voice and Magnitude Spectrum"""
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axes[0, 0].set_title("Original Voice")
axes[0, 0].plot(time,wave, color='red')
axes[0, 1].set_title("Encrypted Voice")
axes[0, 1].plot(time,encr, color='orange')
axes[0, 2].set_title("Decrypted Voice")
axes[0, 2].plot(time,decr, color='green')
axes[1, 0].set_title("Magnitude Spectrum Original Voice")
axes[1, 0].magnitude_spectrum(wave, Fs=2, color='red')
axes[1, 1].set_title("Magnitude Spectrum Encrypted Voice ")
axes[1, 1].magnitude_spectrum(encr, Fs=2, color='orange')
axes[1, 2].set_title("Magnitude Spectrum Decrypted Voice")
axes[1, 2].magnitude_spectrum(decr, Fs=2, color='green')
fig.tight_layout()
#plt.savefig('choas voice encryption.png', dpi=1200)
plt.show()

"""Ploting Portion for Power Spectrum Density and Spectrogram"""
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axes[0, 0].set_title("Power Spectrum Density Original Voice")
axes[0, 0].psd(wave, Fs=2, color='red',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[0, 1].set_title("Power Spectrum Density Encrypted Voice ")
axes[0, 1].psd(encr, Fs=2, color='orange',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[0, 2].set_title("Power Spectrum Density Decrypted Voice")
axes[0, 2].psd(decr, Fs=2, color='green',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[1, 0].set_title("Spectrogram Original Voice")
axes[1, 0].specgram(wave, Fs=4)
axes[1, 1].set_title("Spectrogram Original Encrypted Voice ")
axes[1, 1].specgram(encr, Fs=4)
axes[1, 2].set_title("Spectrogram Original Decrypted Voice")
axes[1, 2].specgram(decr, Fs=4)
fig.tight_layout()
#plt.savefig('choas voice encryption.png', dpi=1200)
plt.show()

