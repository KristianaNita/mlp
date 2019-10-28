import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    opts = parser.parse_args()

    for file in opts.files:
        print(file)
        with open(file, 'r') as f:
            j = json.load(f)
            print(j)
            epochs, _, _, _, valid_acc, _, _ = zip(*j)
            print(epochs, valid_acc)
            plt.plot(epochs, valid_acc, '-', label=file.split('/')[-1].replace('_stats.txt', ''), markersize=3)

    plt.xlim([0, 20])
    plt.xticks(np.arange(0, 21, 2))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy on validation set')
    plt.legend()
    plt.tight_layout()
    plt.show()
