import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
import random
data_path = 'data/'


#labels = [f.split('.')[0] for f in listdir(data_path) if isfile(join(data_path, f))]
def make_column(column_name,mylist,dataset):
    se = pd.Series(mylist)
    dataset[column_name] = se.values
    
def make_data(filenames):
    write_list = []
    for file_name in filenames:
        with open(data_path+file_name, 'rb') as f:
            result = f.read().splitlines()
        label = file_name.split('.')[0]
        for i in result:
            write_list.append([i.decode("utf-8"),label])
    for i in range(10):
        random.shuffle(write_list)
    with open("input.csv", "w") as filewrite:
        writer = csv.writer(filewrite)
        writer.writerows(write_list)
