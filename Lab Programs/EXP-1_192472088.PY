import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [10,20,15,25,30]

plt.plot(x, y, marker='o')
plt.title("Simple Line Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()