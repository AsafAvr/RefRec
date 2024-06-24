import logging
from config import Config
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

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
    for i in tqdm(range(len(data_set)), desc="dataset to dataframe"):
        data = data_set[i]
        user_key, user_idx, user_hist = get_user_hist(data['hist_aug_vec'], hist_idxes)
        item_key, item_idx, item_desc = get_item_desc(data['item_aug_vec'], item_idxes)
        user_id = id2user[str(user_key)]
        user_prompt = user_prompt_dict[user_id]
        item_prompt = item_prompt_dict[item_key]
        data_list.append({'test_index': i, 'user_idx': user_idx, 'user_key': user_key, 'user_hist': user_hist, 'item_idx': item_idx,'item_key': item_key, 'item_desc': item_desc, 'user_prompt': user_prompt, 'item_prompt': item_prompt})
    df = pd.DataFrame(data_list)
    return df

def update_val_new_profile(df_train,df_val,column_name):
    df_train_unique = df_train.drop_duplicates(subset='user_idx')[['user_idx', column_name]]

    # Merge df_val with the unique train data
    df_val = df_val.merge(df_train_unique, on='user_idx', how='left')

    return df_val

def get_model_path():
    model_path = rs_folder.joinpath('model').joinpath('ml-1m').joinpath('ctr').joinpath('DIN').joinpath('DIN.pt').as_posix()
    return model_path

 # Load data
def load_datasets(return_df = True, create_small_sets=False):
    train_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'train', 'ctr', 5, True, 'bert_avg')
    test_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'test', 'ctr', 5, True, 'bert_avg')
    
    if create_small_sets:
        small_train_set, small_val_set = create_small_train_val_sets(train_set)
        train_loader = Data.DataLoader(dataset=small_train_set, batch_size=32, shuffle=False)
        val_loader = Data.DataLoader(dataset=small_val_set, batch_size=32, shuffle=False)
    else:
        train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=False)
    
    if return_df:
        if create_small_sets:
            df_small_train = data_set_to_dataframe(small_train_set, hist_idxes, item_idxes)
            df_small_val = data_set_to_dataframe(small_val_set, hist_idxes, item_idxes)
            return df_small_train, small_train_set, train_loader, df_small_val, small_val_set, val_loader
        else:
            df_train = data_set_to_dataframe(train_set, hist_idxes, item_idxes)
            train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=False)

            df_test = data_set_to_dataframe(test_set, hist_idxes, item_idxes)
            test_loader = Data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)
            return df_train, train_set, train_loader, df_test, test_set, test_loader
    else:
        if create_small_sets:
            return small_train_set, train_loader, small_val_set, val_loader, test_set, test_loader
        else:
            df_train = data_set_to_dataframe(train_set, hist_idxes, item_idxes)
            train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=False)
            df_test = data_set_to_dataframe(test_set, hist_idxes, item_idxes)
            test_loader = Data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)
            return train_set, train_loader, test_set, test_loader

def load_full_datasets():
    train_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'train', 'ctr', 5, True, 'bert_avg')
    test_set = AmzDataset(ml1m_folder.joinpath('proc_data').as_posix(), 'val', 'ctr', 5, True, 'bert_avg')
    train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=2, shuffle=False)
    df_train = data_set_to_dataframe(train_set, hist_idxes, item_idxes)
    df_test = data_set_to_dataframe(test_set, hist_idxes, item_idxes)
    return df_train, train_set, train_loader, df_test, test_set, test_loader

def dataset_to_df_loader(dataset):
    df = data_set_to_dataframe(dataset, hist_idxes, item_idxes)
    loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    return df, loader

def load_local_data(train_path, val_path):
    with open(train_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_set = pickle.load(f)
    df_train, train_loader = dataset_to_df_loader(train_set)
    df_val, val_loader = dataset_to_df_loader(val_set)
    return df_train, train_set, train_loader, df_val, val_set, val_loader

def create_small_train_val_sets(train_set, val_ratio=0.2, num_users=100, sample_size=20):
    import random
    
    logging.info(f"Creating small train and validation sets with {num_users} users and {val_ratio:.0%} validation ratio")
    
    # Get unique users, stop when num_users unique users are found
    unique_users = []
    for data in train_set:
        if data['uid'] not in unique_users:
            unique_users.append(data['uid'])
        if len(unique_users) == num_users:
            break
    
    # Randomly select a subset of users
    selected_users = random.sample(unique_users, min(num_users, len(unique_users)))
    
    train_data = []
    val_data = []
    
    for user in tqdm(selected_users, desc="Processing users"):
    
        user_data = []
        for data in train_set:
            if data['uid'] == user:
                user_data.append(data)
            if len(user_data) > sample_size:
                break
        # Split user data into train and validation without shuffling
        split_index = int(len(user_data) * (1 - val_ratio))
        train_data.extend(user_data[:split_index])
        val_data.extend(user_data[split_index:])
    
    logging.info(f"Created small train set with {len(train_data)} entries")
    logging.info(f"Created small validation set with {len(val_data)} entries")
    
    return DictDataset(train_data), DictDataset(val_data)

def print_dataset_info(df):
    num_distinct_users = df['user_idx'].nunique()
    logging.info(f"Number of distinct users: {num_distinct_users}")
    logging.info(f"Length of dataset: {len(df)}")

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
