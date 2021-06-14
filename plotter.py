import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def smoothen(x, k):
    n = x.shape[0]
    y = []
    if k >= 0:
        sum_v = 0
        for i in range(k):
            sum_v += x[i]
            y.append(sum_v/(i+1))
        for i in range(k, n):
            sum_v += x[i]
            sum_v -= x[i-k]
            y.append(sum_v/k)
    return np.array(y)


#Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--average", default=0, type=int)
parser.add_argument("--y1", default="invalid")
parser.add_argument("--y2", default="invalid")
parser.add_argument("--labels", default="invalid label")
parser.add_argument("--twin_ax", default="F")
args = parser.parse_args()

if args.y1 == "invalid":
    print("pass atleast one array to print via --y1")
    sys.exit(0)

y1 = np.load(args.y1)
if args.average>0:
    y1 = smoothen(y1, args.average)
if args.y2 != "invalid":
    y2 = np.load(args.y2)
    if args.average>0:
        y2 = smoothen(y2, args.average)

labels = args.labels.split()
#define font size
plt.rcParams.update({'font.size': 15})

n = min(y1.shape[0], y2.shape[0])
y1 = y1[:n]
y2 = y2[:n]
x = [i for i in range(1, n+1)]

fig, ax1 = plt.subplots()
ax1.set_xlabel('epochs')

color = 'tab:red'
ax1.set_ylabel('Reward', color='black')  # we already handled the x-label with ax1
ax1.plot(x, y1, label = labels[0], color=color)
ax1.tick_params(axis='y', labelcolor='black')

if args.y2 != "invalid":
    color = 'tab:blue'
    if args.twin_ax == "F":
        ax1.plot(x, y2, label = labels[1], color=color)
    else:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Reward2', color=color)
        ax2.plot(x, y2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

#for displaying grids
plt.grid()

#for finely defining where to place values on x-axis
plt.xticks([int(n/4), int(n/2), int((3*n)/4), n])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
if args.labels != "invalid label":
    plt.legend(loc='upper left')
plt.show()
