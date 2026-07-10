import pandas as pd
from sklearn.utils import shuffle
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


data =pd.read_csv("./dataset/a_input_pro.csv",index_col=None,header=0, encoding='utf-8')
data = shuffle(data)
data.columns=["smiles","pIC50(M)"]
data.to_csv('./data/a_input_pro/raw/a_input_plus_random.csv', index=False)

dataset_train, dataset_test= split_dataset(data, 0.8)
dataset_train=pd.DataFrame(dataset_train)
dataset_test=pd.DataFrame(dataset_test)

dataset_train.columns=["smiles","label"]
dataset_test.columns=["smiles","label"]
dataset_train.to_csv("./datashuffle/a_input_plus_ramdom_train.csv", index=False)
dataset_test.to_csv("./datashuffle/a_input_plus_ramdom_test.csv", index=False)

