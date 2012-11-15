import matplotlib.pyplot as pyplot

sigma_squared = [0.0, 0.5, 1.0, 1.5, 2.0]

aucs = []
for line in open("aucs.txt", "r"):
	aucs.append(float(line.rstrip("\n")))

fig = pyplot.figure()
ax = fig.add_subplot(111)
pyplot.title("AUC vs. Noise Level")

ax.set_xlabel('Noise Level')
ax.set_ylabel('AUC')

ax.plot(sigma_squared, aucs)

fontP = FontProperties()
fontP.set_size('small')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),  prop = fontP)

fig.savefig("aucs.png");
