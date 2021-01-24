import matplotlib.pyplot as plt

# Results were taken from "Deep Residual Learning for Image Recognition"

x = ["ILSVRC'10", "ILSVRC'11", "ILSVRC'12\nAlexNet", "ILSVRC'13", "ILSVRC'14\nVGG", "ILSVRC'14 \nGoogLeNet", "ILSVRC'15\nResNet"]
y = [28.2, 25.8, 16.4, 11.7, 7.3, 6.7, 3.57]
num_of_layers = [1.5, 2.5, 7.5, 8.5, 19, 22, 152]


fig, ax1 = plt.subplots()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.get_yaxis().set_visible(False)
ax1.bar(x[:-1], y[:-1], width=0.35, color='#436bb5')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.get_yaxis().set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.plot(x[:-1], num_of_layers[:-1], color='orange', linestyle=':', marker='v')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

fig, ax1 = plt.subplots()
ax1.get_yaxis().set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.bar(x, y, width=0.35, color='#436bb5')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.get_yaxis().set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.plot(x, num_of_layers, color='orange', linestyle=':', marker='v')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
