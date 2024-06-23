import logging
from config import Config
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

cur_path = Path.cwd()
klg_folder = Path.cwd().joinpath('knowledge_encoding')
rs_folder = Path.cwd().joinpath('RS')
ml1m_folder = Path.cwd().joinpath('data').joinpath('ml-1m')
proc_folder = ml1m_folder.joinpath('proc_data')
preprocess_folder = Path.cwd().joinpath('preprocess')

import sys
sys.path.append(klg_folder.as_posix())
sys.path.append(rs_folder.as_posix())
sys.path.append(preprocess_folder.as_posix())

from lm_encoding import get_text_data_loader, inference
from pre_utils import GENDER_MAPPING, AGE_MAPPING, OCCUPATION_MAPPING
from dataset import AmzDataset
from utils import load_json, load_pickle

class DictDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


user_vec_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('bert_avg_augment.hist'))
item_vec_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('bert_avg_augment.item'))

user_prompt_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('prompt.hist').as_posix())
item_prompt_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('prompt.item').as_posix())

hist_loader, hist_idxes, item_loader, item_idxes = get_text_data_loader(ml1m_folder.joinpath('knowledge').as_posix(), 1)

user_prompt_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('prompt.hist').as_posix())
item_prompt_dict = load_json(ml1m_folder.joinpath('proc_data').joinpath('prompt.item').as_posix())
datamap = load_json(ml1m_folder.joinpath('proc_data').joinpath('datamaps.json').as_posix())
sequence_data = load_json(ml1m_folder.joinpath('proc_data').joinpath('sequential_data.json').as_posix())
train_test_split = load_json(ml1m_folder.joinpath('proc_data').joinpath('train_test_split.json').as_posix())
lm_hist_idx = train_test_split['lm_hist_idx']
id2user = datamap['id2user']
id2item = datamap['id2item']
itemid2title = datamap['itemid2title']
user2attribute = datamap['user2attribute']

def return_user_details(uid):
    user = id2user[uid]
    item_seq, rating_seq = sequence_data[uid]
    cur_idx = lm_hist_idx[uid]
    hist_item_seq = item_seq[:cur_idx]
    hist_rating_seq = rating_seq[:cur_idx]
    history_texts = []
    movie_history = []
    for iid, rating in zip(hist_item_seq, hist_rating_seq):
        movie_history.append({'title': itemid2title[str(iid)], 'rating': rating})
    gender, age, occupation = user2attribute[uid]
    gender = GENDER_MAPPING[gender]
    age = AGE_MAPPING[age]
    occupation = OCCUPATION_MAPPING[occupation]
    return gender, age, occupation, movie_history

def get_user_hist(user_vec,hist_idxes):
    user_vec = user_vec.tolist()
    for key, v in user_vec_dict.items():
        if v == user_vec:
            user_index = hist_idxes.index(key)
            return key,user_index,hist_loader.dataset[user_index]
    return None

def get_item_desc(item_vec, item_idxes):
    item_vec = item_vec.tolist()
    for key,v in item_vec_dict.items():
        if v == item_vec:
            item_index = item_idxes.index(key)
            return key,item_index, item_loader.dataset[item_index]
    return None

def data_set_to_dataframe(data_set, hist_idxes, item_idxes):
    data_list = []
    for i in tqdm(range(len(data_set))):
        data = data_set[i]
        user_key, user_idx, user_hist = get_user_hist(data['hist_aug_vec'], hist_idxes)
        item_key, item_idx, item_desc = get_item_desc(data['item_aug_vec'], item_idxes)
        user_id = id2user[str(user_key)]
        user_prompt = user_prompt_dict[user_id]
        item_prompt = item_prompt_dict[item_key]
        data_list.append({'test_index': i, 'user_idx': user_idx, 'user_key': user_key, 'user_hist': user_hist, 'item_idx': item_idx,'item_key': item_key, 'item_desc': item_desc, 'user_prompt': user_prompt, 'item_prompt': item_prompt})
    df = pd.DataFrame(data_list)
    return df

def get_model_path():
    model_path = rs_folder.joinpath('model').joinpath('ml-1m').joinpath('ctr').joinpath('DIN').joinpath('DIN.pt').as_posix()
    return model_path

 # Load data
def load_datasets(return_df = True):
    # user_ids_list = [1954,4621,5470,1461,4597,2050,1902,4434,5264,5880,2336]
    train_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'train', 'ctr', 5, True, 'bert_avg')#, user_ids=user_ids_list) #test
    test_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'test', 'ctr', 5, True, 'bert_avg')#, user_ids=user_ids_list) #test
    
    train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    if return_df:
        df_train = data_set_to_dataframe(train_set, hist_idxes, item_idxes)
        df_test = data_set_to_dataframe(test_set, hist_idxes, item_idxes)
        return df_train,train_set, train_loader, df_test, test_set, test_loader
    else:
        return train_set, train_loader, test_set, test_loader

def update_dataset_aug_vec(df, user_col_name, dataset, encoding_model, tokenizer):
    #dataset to list
    data_list = [dataset[i] for i in range(len(dataset))]
    
    new_user_history = df[user_col_name].tolist()
    data_l = DataLoader(new_user_history,2, shuffle=False)
    new_user_vec = inference(encoding_model, tokenizer, data_l, 'bert', 'avg')

    for idx, item in enumerate(data_list):
        item['hist_aug_vec'] = torch.tensor(new_user_vec[idx])

    new_dataset = DictDataset(data_list)
    new_dataloader = DataLoader(new_dataset, 32, shuffle=False)
    return new_dataset, new_dataloader


