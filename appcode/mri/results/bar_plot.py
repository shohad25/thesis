import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('ggplot')

# psnr_no_masking = {
#     "2.5": {
#         "Zero-filled": [32.044, 2.616],
#         "CS-MRI": [39.053, 2.479],
#         "IM-CNN-L2": [35.281, 2.210],
#         "CNN-L2": [38.978, 2.454],
#         "Proposed": [39.802, 2.489],
#     },
#     "4": {
#         "Zero-filled": [26.447, 1.997],
#         "CS-MRI": [33.273, 2.452],
#         "IM-CNN-L2": [31.577, 2.109],
#         "CNN-L2": [33.405, 2.232],
#         "Proposed": [34.595, 2.519],
#     },
#     "6": {
#         "Zero-filled": [16.470, 2.181],
#         "CS-MRI": [26.951, 3.380],
#         "IM-CNN-L2": [23.417, 2.676],
#         "CNN-L2": [31.010, 2.299],
#         "Proposed": [31.555, 2.487],
#     }
# }

ssim_no_masking = {
    "2.5": {
        "Zero-filled": [0.6984, 0.0452],
        "CS-MRI": [0.8650, 0.0297],
        "CNN-L2": [0.8859, 0.0307],
        "IM-CNN-L2": [0.8849, 0.0307],
        "Proposed": [0.91771, 0.0199],
    },
    "4": {
        "Zero-filled": [0.5907, 0.040],
        "CS-MRI": [0.7574, 0.0376],
        "CNN-L2": [0.7492, 0.0434],
        "IM-CNN-L2": [0.7967, 0.036],
        "Proposed": [0.8184, 0.034],
    },
    "6": {
        "Zero-filled": [0.2550, 0.0218],
        "CS-MRI": [0.6179, 0.0301],
        "CNN-L2": [0.6820, 0.0423],
        "IM-CNN-L2": [0.606, 0.0231],
        "Proposed": [0.7264, 0.0388],
    }
}

psnr_no_masking = {
    "2.5": {
        "Zero-filled": [32.555, 2.294],
        "CS-MRI": [38.782, 1.750],
        "CNN-L2": [39.394, 1.985],
        "IM-CNN-L2": [39.663, 1.934],
        "Proposed": [40.211, 1.902],
    },
    "4": {
        "Zero-filled": [26.791, 1.856],
        "CS-MRI": [33.088, 1.929],
        "CNN-L2": [33.829, 2.034],
        "IM-CNN-L2": [35.041, 2.042],
        "Proposed": [35.133, 1.870],
    },
    "6": {
        "Zero-filled": [16.781, 1.737],
        "CS-MRI": [26.491, 2.587],
        "CNN-L2": [31.403, 2.040],
        "IM-CNN-L2": [28.134, 2.181],
        "Proposed": [32.040, 2.110],
    }
}

# example data
methods = psnr_no_masking['2.5'].keys()
x = np.array([2.5, 4, 6])
y_mean = {}
y_std = {}
for method in methods:
    y_mean[method] = [psnr_no_masking["2.5"][method][0], psnr_no_masking["4"][method][0], psnr_no_masking["6"][method][0]]
    y_std[method]  = [0.5*psnr_no_masking["2.5"][method][1], 0.5*psnr_no_masking["4"][method][1], 0.5*psnr_no_masking["6"][method][1]]

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
fmts = ['D', '+', '|', 'x', 'o']
ind = 0
ad = 0
for method in methods:
    plt.errorbar(x+ad, y_mean[method], yerr=y_std[method], fmt=fmts[methods.index(method)], label=method, capsize=5, capthick=1.5)
    ad += 0.05

plt.title("Error in PSNR, without masking")
plt.grid('on', linestyle='-', linewidth=0.2)
plt.ylabel('PSNR[dB]')
plt.xlabel('Sampling factor')
plt.legend()
plt.xticks(x, x)
plt.show()