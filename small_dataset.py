import pickle
import random
from collections import defaultdict
from data_utils import load_full_datasets, print_dataset_info


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def create_small_datasets(input_file, output_path, num_users=100):
    # Load the original data
    data = load_pickle(input_file)

    # Organize data by user
    user_data = defaultdict(list)
    for entry in data:
        uid = entry[0]
        user_data[uid].append(entry)

    # Select random users
    selected_users = random.sample(list(user_data.keys()), num_users)

    # Create new datasets
    new_train, new_val, new_test = [], [], []

    for uid in selected_users:
        entries = sorted(user_data[uid], key=lambda x: x[1])  # Sort by seq_idx
        if len(entries) >= 10:
            new_test.append(entries[-1])
            new_val.extend(entries[-6:-1])
            new_train.extend(entries[:-6])


    # Save new datasets
    save_pickle(new_train, f'{output_path}/ctr.train')
    save_pickle(new_val, f'{output_path}/ctr.val')
    save_pickle(new_test, f'{output_path}/ctr.test')

    print(f"Created new datasets with {len(new_train)} train, {len(new_val)} val, and {len(new_test)} test entries.")
    print(f"Number of unique users: {len(selected_users)}")

if __name__ == "__main__":
    input_file = '/home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/data/ml-1m/proc_data/ctr.test_org'
    output_path = '/home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/temp'
    create_small_datasets(input_file, output_path)

    df_train, train_set, train_loader, df_test, test_set, test_loader = load_full_datasets()

    print_dataset_info(df_train)
    print_dataset_info(df_test)