#!/usr/bin/env python3

'''
!pip3 install -U tensorflow-gpu keras numpy scipy
'''

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.linalg.blas import daxpy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, GRU
from keras.models import Sequential
from keras.optimizers import RMSprop


# Load a short audio file
(breath_sr, breath_wav) = wavfile.read('11_Dialogue_Class_Rank.wav')
# Retain just one stereo track
data = breath_wav[:,0]

# GPUs to use (0-1)
gpu_count = 1
# Number of time-domain samples to convert to frequency domain in one window
window_size = 1024
# Number of time-domain samples between successive windows
slide = 256
# Frequency domain dimension
freq_dim = 2 * (1 + (window_size // 2)) # x2 because of real-imag parts
# Number of successive freq domain windows to predict the next window from
sequence_len = 25
# Dimension of GRU units
gru_dim = 1024
# Optimizer learning rate
learning_rate=0.1

specgram = plt.specgram(data, NFFT=window_size, Fs=slide)
print("Spectrum of input audio")
plt.show()

# Hanning window weights to apply to time-domain sample windows
# Normalize weights to sum to 1, for later convenience
window_weight = slide * np.hanning(window_size) / np.sum(np.hanning(window_size))
n = len(data)

# Data, sliced into a series of windows, and weighted
weighted_slices = data[np.arange(window_size)[None, :] + slide * np.arange(1 + (n - window_size) // slide)[:, None]] * window_weight
del data
# Apply the FFT to convert to a sequence of frequency-domain windows
freq_slices = np.fft.rfft(weighted_slices)
del weighted_slices
# FFT outputs (real,imag) 64-bit pairs. Flatten them to two separate 32-bit values
freq_slices_flattened = np.apply_along_axis(lambda a: a.view('(2,)float').flatten(), 1, freq_slices).astype('float32')
del freq_slices

# Select devices for training based on GPU availability
if gpu_count > 0:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    K.set_session(session)
    device1 = '/gpu:0'
    if gpu_count > 1:
        device2 = '/gpu:1'
    else:
        device2 = '/gpu:0'
else:
    device1 = '/cpu:0'
    device2 = '/cpu:0'

model = Sequential()
with tf.device(device1):
    model.add(GRU(gru_dim, 
                  input_shape=(sequence_len, freq_dim)))
with tf.device(device2):
    model.add(Dense(freq_dim, activation=None))

model.compile(optimizer=RMSprop(lr=learning_rate), 
              loss='mean_absolute_error')
model.summary()

# Initialize predicted audio out with the first few input windows
predicted_freq_slices = freq_slices_flattened[:sequence_len]

# Build up batches of input windows, paired with next window as prediction target 
input_freq_slices = []
next_freq_slices = []
for i in range(0, len(freq_slices_flattened) - sequence_len - 1):
    input_freq_slices.append(freq_slices_flattened[i : i + sequence_len])
    next_freq_slices.append(freq_slices_flattened[i + sequence_len])
del freq_slices_flattened

# Convert them to numpy arrays for future use
input_freq_slices = np.array(input_freq_slices)
next_freq_slices = np.array(next_freq_slices)

# Pick most (input,next) pairs as training; rest for validation
shuffled_indices = np.random.permutation(len(input_freq_slices))
training_size = int(0.95 * len(input_freq_slices))
train_indices = shuffled_indices[:training_size]
val_indices = shuffled_indices[training_size:]

input_freq_slices_train = input_freq_slices[train_indices]
input_freq_slices_val = input_freq_slices[val_indices]
next_freq_slices_train = next_freq_slices[train_indices]
next_freq_slices_val = next_freq_slices[val_indices]

early_stopping = EarlyStopping(patience=10, verbose=1)

model.fit(input_freq_slices_train,
          next_freq_slices_train,
          epochs=100,
          batch_size=64,
          shuffle=True,
          validation_data=(input_freq_slices_val, next_freq_slices_val),
          verbose=2,
          callbacks=[early_stopping])

# Starting with initial part of input audio, predict many next windows
for i in range(0, 1000):
    pred_next_slice = model.predict(predicted_freq_slices[None,-sequence_len:])
    predicted_freq_slices = np.append(predicted_freq_slices, pred_next_slice, axis=0)

# Convert back to (real,imag) complex representation in freq domain
predicted_freq_slices_unflattened = \
     np.reshape(predicted_freq_slices, (-1, freq_dim//2, 2)).view('complex64').reshape(-1, freq_dim//2).astype('complex128')
  
# Apply inverse FFT to get back time-domain windows
pred_time_slices = np.fft.irfft(predicted_freq_slices_unflattened)

# Reassemble full time domain signal by adding overlapping windows
reassembled = np.zeros(window_size + (len(pred_time_slices) - 1) * slide)
for i in range(0, len(pred_time_slices)):    
    daxpy(pred_time_slices[i], reassembled, offy=slide * i)
  
# Plot some of the first generated time-domain data as a check
plot_sample_base = sequence_len * slide
plt.plot(reassembled[plot_sample_base:plot_sample_base + window_size])
plt.show()
  
# Scale time-domain data to have max at 32767, for 16-bit wav output
reassembled_scale = np.max(np.abs(reassembled))
reassembled = reassembled * (32767 / reassembled_scale)

print("Spectrum of output audio")
specgram = plt.specgram(reassembled, NFFT=window_size, Fs=slide)
plt.show()

# Overwrite output to out.wav
out_file = 'out.wav'
if os.path.isfile(out_file):
    os.remove(out_file)
wavfile.write(out_file, breath_sr, reassembled.astype(np.int16))
