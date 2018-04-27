import matplotlib.pyplot as plt
import numpy as np

def plot_bar(x, y, save_name):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    major_ticks = np.arange(0, 101, 0.1)
    minor_ticks = np.arange(0, 101, 0.05)
    
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    # And a corresponding grid
    ax.grid(which='both')
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)
    
    # plot the data, ensuring the ytick is setup with a decent step size.
    plt.bar(x, y, 1/1.5, align='center')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(True, drawstyle="steps")
    
    # set the labels and the title of the plot.
    plt.xlabel("Taggers")
    plt.ylabel("Accuracy Achieved")
    plt.title("Average Accuracy Achieved by Taggers on the Brown Corpus")
    
    # save the plotting to a local file.
    fig.set_size_inches(8, 5)
    fig.savefig("images/" + save_name + ".png", dpi=250, bbox_inches='tight')
    
def plot(x, y, save_name):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    major_ticks_x = np.arange(0, 35000, 5000)
    minor_ticks_x = np.arange(0, 35000, 1000)
    major_ticks_y = np.arange(0, 1.1, 0.1)
    minor_ticks_y = np.arange(0, 1.1, 0.05)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # And a corresponding grid
    ax.grid(which='both')
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)
    
    plt.xlabel("Size of Training Data Provided")
    plt.ylabel("Accuracy Achieved")
    plt.title("Unigram Accuracy Given Varying Training Sizes")
    
    # plot the data, ensuring the ytick is setup with a decent step size.
    plt.plot(x, y, ":ro")
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    # save the plotting to a local file.
    fig.set_size_inches(8, 5)
    fig.savefig("images/" + save_name + ".png", dpi=250, bbox_inches='tight')