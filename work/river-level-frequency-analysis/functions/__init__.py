import csv
import dateutil
import numpy as np


def read_levels_file(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f, delimiter=","))

    dates = [dateutil.parser.isoparse(r[0]) for r in data[1:]]
    levels = np.array([float(r[1]) for r in data[1:]])
    return (dates, levels)
    
def fft_analysis(levels, fs, fft_size):
    overlap_fac=0.5
    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.floor(len(levels) / np.float32(hop_size)))
    t_max = len(levels) / np.float32(fs)
    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
    
    proc = np.concatenate((levels, np.zeros(pad_end_size)))              # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)      # space to hold the result
    for i in range(total_segments):                       # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i, :] = autopower[:fft_size]               # append to the results array

    result = 20*np.log10(result[:-1])    # scale to db, dropping the last value
    result = np.clip(result, -165, 200)    # clip values

    return result