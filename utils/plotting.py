def plot_data(data):
    import matplotlib.pyplot as plt
    shape = data.shape
    sample_data = data
    dataX = []
    dataY = []

    for i in range(shape[0]):
        for time in range(shape[1]):
            for pitch in range(shape[2]):
                if sample_data[i][time][pitch] > 0.1:
                    dataX.append(time + i * shape[1])
                    dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()