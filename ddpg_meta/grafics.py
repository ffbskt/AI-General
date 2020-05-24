import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

DIV_LINE_WIDTH = 10

# Global vars for tracking and labeling data at load time.
os.chdir("./data_log")
dt = None
for file in glob.glob("*.csv"):
    if dt is None:

        print(file)
        dt = pd.read_csv(file)
        dt['agent name'] = 'a' + str(file[:-4]) #
        #print(dt.head())
    else:
        ndt = pd.read_csv(file)
        ndt['agent name'] = 'a' + str(file[:-4])
        dt = dt.append(ndt, ignore_index=True)

    print(dt[dt['TotalEnvInteracts']==50])

sns.lineplot(x='TotalEnvInteracts', y='AverageEpRet',  hue='agent name', data=dt)#, palette=[1])

    #plt.legend()
plt.show()
