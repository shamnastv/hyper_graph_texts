import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 22})
axes = plt.gca()

# Val Prop
# color = 'blue'
# x = ['2.5%', '5%', '25%', '50%', '75%']
# y1 = [0.9221, .9321, .9626, .9685, .9707]
# y2 = [0.44583, 0.51521, 0.65286, 0.6910, 0.7001]
# axes.set_ylim([.3, 1.0])
#
# plt.xlabel('Training Percentage')
# plt.scatter(x, y1, label='R8', color=color)
# plt.plot(x, y1, color=color)
# plt.scatter(x, y2, label='Ohsumed', color='red')
# plt.plot(x, y2, color='red')
# plt.ylabel('Accuracy')
# plt.rcParams.update({'font.size': 14})
# plt.legend(loc='lower right')
# # plt.legend()
# plt.grid()
# plt.show()
#
#
# Layers
color = 'blue'
x = ['1', '2', '3', '4']
y1 = [0.9708, .9743, 0.96849, 0.9655]
y2 = [0.71172, 0.7238, 0.71259, 0.7023]
axes.set_ylim([.6, 1.0])

plt.xlabel('Number of Layers')
plt.scatter(x, y1, label='R8', color=color)
plt.plot(x, y1, color=color)
plt.scatter(x, y2, label='Ohsumed', color='red')
plt.plot(x, y2, color='red')
plt.ylabel('Accuracy')
plt.rcParams.update({'font.size': 14})
plt.legend(loc='lower right')
# plt.legend()
plt.grid()

plt.show()
#
# # Hidden Dime
# color = 'blue'
# x = ['50', '100', '200', '300']
# y1 = [0.9678, 0.9691, .9743, 0.9663]
# y2 = [0.7032, 0.7110, .7238, 0.7158]
#
# axes.set_ylim([.6, 1.0])
#
# plt.xlabel('Hidden dimension')
# plt.scatter(x, y1, label='R8', color=color)
# plt.plot(x, y1, color=color)
# plt.scatter(x, y2, label='Ohsumed', color='red')
# plt.plot(x, y2, color='red')
# plt.ylabel('Accuracy')
# plt.rcParams.update({'font.size': 14})
# plt.legend(loc='lower right')
# # plt.legend()
# plt.grid()
#
# plt.show()

#
# Clusters
color = 'blue'
x = ['1', '2', '3', '4', '8', '16', '32']
# plt.xscale('log', basex=2)
# x = [1, 2, 4, 8, 16, 32]
y1 = [0.9725, 0.9743, 0.9689, 0.96849, 0.96758, 0.96804, 0.96598]
y2 = [0.7196, 0.7223, 0.7238, 0.7122, 0.7156, 0.7151, 0.7201]
# plt.xticks([1, 2, 3, 4, 8, 16, 32, 64])
axes.set_ylim([.6, 1.0])

plt.xlabel('Number of Clusters')
plt.scatter(x, y1, label='R8', color=color)
plt.plot(x, y1, color=color)
plt.scatter(x, y2, label='Ohsumed', color='red')
plt.plot(x, y2, color='red')
plt.ylabel('Accuracy')
plt.rcParams.update({'font.size': 14})
plt.legend(loc='lower right')
# plt.legend()
plt.grid()

plt.show()
