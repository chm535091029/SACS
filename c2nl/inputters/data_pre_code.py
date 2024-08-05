import json
from tqdm import tqdm

def read_json_code(json_file,save_file):
    f = open(json_file,'r')
    f1 = open(save_file,"w",encoding="utf-8")
    dataset = f.readlines()
    for data in tqdm(dataset):
        data = json.loads(data)
        f1.write(data['code'].replace('\n','').replace('\r','')+'\n')

    f.close()
    f1.close()

read_json_code("../../data/java/train.json","../../data/java/train3.source.code")
read_json_code("../../data/java/valid.json","../../data/java/valid3.source.code")
read_json_code("../../data/java/test.json","../../data/java/test3.source.code")