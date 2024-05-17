import pandas as pd
import os
from noaa_sdk import NOAA

def get_data():
    n = NOAA()
    observations = n.get_observations('11365','US')
    for observation in observations:
        print(observation)
        break

    df = pd.DataFrame(observations)
    df.to_csv(r".\data.csv", index=False)

def dvc_push():
    # Change directory to project root (where DVC is initialized)
    # os.chdir(r'F:\University Material\Semester VIII\Mlops\CA7')
    # os.system('cd ..')
    # Stage and commit new data with DVC
    os.system('cd ..')
    os.system('dvc add data/data.csv')
    os.system('dvc commit -m "Push updated data.csv"')

    # Push changes to remote storage (if configured)
    os.system('dvc push')

if __name__ == "__main__":
    get_data()
    dvc_push()