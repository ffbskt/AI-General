import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
os.chdir("./data_log")
for file in glob.glob("*.csv"):
    print(file)
    sns.lineplot(x='TotalEnvInteracts', y='AverageEpRet', hue='agent name', data=pd.read_csv(file))#  # style="event")
    plt.show()