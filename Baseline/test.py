from pywt import wavedec
import pywt
import matplotlib.pyplot as plt


# Signal data is from slide 25 of the Patton_MedResearchOverview_v2 PowerPoint
# that I emailed you earlier this week. The signal was padded with 3 zeros
# on the end in order to have 16 elements
signal = [0,1,0,0,0,0,4,4,4,1,1,0,0,0,0,0]

# Deconstruct the signal TO the Haar wavelet
coeffs = wavedec(signal, 'haar')
# print (len(coeffs))
for x in range(0, len(coeffs)):
    print(x, ": ", coeffs[x])

print ('\n\n')

# Reconstruct the signal FROM the Haar wavelet
# Prior to reconstruction, you can zero out some of the wavelet coefficients
# which is often done to remove noise in the signal or for data compression.
# In these examples, I haven't done that so that you can see that the signal
# is perfecting reconstructed.
signal = pywt.waverec(coeffs, 'haar')
print (signal)

print ('\n\n')


# first 16 elements of the HR column from e058-u-001-h.csv file
signal = [
372.4858503,
902.0385416,
512.3290625,
905.9622013,
797.5304948,
480.7314389,
870.958063,
723.4580476,
656.7657123,
836.3403228,
439.3853287,
1020.611512,
340.7800353,
1006.286453,
450.7889166,
927.5356373
]

# Deconstruct the signal TO the Haar wavelet
coeffs = wavedec(signal, 'haar')
# print (len(coeffs))
for x in range(0, len(coeffs)):
    print(x, ": ", coeffs[x])

print ('\n\n')

# Reconstruct the signal FROM the Haar wavelet
signal = pywt.waverec(coeffs, 'haar')
print (signal)
print('\n \n')

plt.plot(signal)
plt.show()