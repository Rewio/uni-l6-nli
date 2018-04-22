import matplotlib.pyplot as plt
import numpy as np

def plot_bar(x, y, save_name):
    
    fig = plt.figure()
    
    # plot the data, ensuring the ytick is setup with a decent step size.
    plt.bar(x, y, 1/1.5, align='center')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(True)
    
    # set the labels and the title of the plot.
    plt.xlabel("Taggers")
    plt.ylabel("Accuracy Achieved")
    plt.title("Average Accuracy Achieved on each Category of the Brown Corpus")
    
    # save the plotting to a local file.
    fig.set_size_inches(8, 5)
    fig.savefig("images/" + save_name + ".png", dpi=250, bbox_inches='tight')