# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import matplotlib.pyplot as plt
import os

dir= <directory of .out files>

accs=np.genfromtxt(dir+"accs.txt", dtype=float)
times=np.genfromtxt(dir+"times.txt", dtype=float)

times=np.cumsum(times)


plt.figure()
plt.plot(times, accs, '-o', label='init from 32pixels, 5 epochs step, 32 pix increase')
plt.title('Best results 1gpu. Baseline + localvit* + different initial resolutions')
plt.xlabel('time(h)')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()
