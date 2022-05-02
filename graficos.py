import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(score, docs):
    plt.figure(figsize=(10, 9))
    sns.heatmap(score, xticklabels=docs, yticklabels=docs, annot=True)
    plt.show()
