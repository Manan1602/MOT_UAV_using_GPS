import json
import pandas as pd
from tqdm import tqdm

def extract_gps(phase :str):
    with open(f'annotations/instances_{phase}_objects_in_water.json') as f:
        data = json.load(f)
    print(data.keys())
    print(data['info'])
    df = pd.DataFrame.from_dict(data['images'])
    print(df['meta'][0])
    for i in tqdm(range(len(df))):
        df1 = pd.DataFrame(df.loc[i,'meta'], index = [0])
        # df1.to_csv(f'Compressed/{phase}/{df.loc[i,'video_id']}/{df.loc[i,'id']}_gps.txt', index = False)
        break
phase = ['train','val']

df = pd.read_csv('Compressed/val/0/267_gps.txt')
extract_gps('train')