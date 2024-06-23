import torch
import os
import time
import numpy as np
from config import Config
from pathlib import Path
import logging
from dotenv import load_dotenv
from main_ctr import model_eval
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from lm_encoding import inference
from unsloth import FastLanguageModel 
from tqdm import tqdm
import sys
rs_folder = Path.cwd().joinpath('RS')
sys.path.append(rs_folder.as_posix())
from main_ctr import parse_args, load_rec_model, get_optimizer

# load_dotenv(Path.cwd().joinpath('.env'))

def load_encoder_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',  trust_remote_code=True)
    encoding_model = AutoModel.from_pretrained('bert-base-uncased',  trust_remote_code=True).half().cuda()
    return tokenizer, encoding_model

def encoding_model_inference(df, encoding_model, tokenizer):
    items = df['item_desc'].tolist()
    item_loader = DataLoader(items, 1, shuffle=False)
    new_item_vec = inference(encoding_model, tokenizer, item_loader, 'bert', 'avg')

    users = df['user_hist'].tolist()
    user_loader = DataLoader(users, 1, shuffle=False)
    new_user_vec = inference(encoding_model, tokenizer, user_loader, 'bert', 'avg')

    return new_item_vec, new_user_vec

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_model = torch.load(model_path,map_location=torch.device(device))
    logging.info(f"Model {model_path} loaded on {device}")
    return inference_model

def evaluate_model(model, data_loader, df, column_name='preds', subset = 'train'):
    auc, ll, loss, eval_time, labels, preds = model_eval(model, data_loader) #short_loader
    logging.info(f"{subset} loss: {loss}, inference time: {eval_time}, auc: {auc}, logloss: {ll}")

    df['labels'] = labels
    df[column_name] = [pred[0] for pred in preds]
    logging.info(f"Model evaluation on {subset} with {column_name} completed successfully.")
    return df

def load_llm(model_name = "unsloth/llama-3-8b-Instruct"):
    hf = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 1000},

    )
    logging.info("LLM loaded successfully.")

    return hf

def load_llm_unsloth(max_seq_length = 2500):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct",
        max_length = max_seq_length,
        )

    FastLanguageModel.for_inference(model) 
    logging.info("LLM loaded successfully.")

    return model, tokenizer
    
def generate_text(model,tokenizer,prompt,max_seq_length = 2500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_seq_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def llm_prompt(prompt_text,llm):
    system = "You are a helpful assistant in the movie recommendation domain."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm
    initial = chain.invoke({"text": prompt_text}).content
    return initial

def add_new_prompt_to_df(df, llm, tokenizer):
    unique_user_ids = df['user_idx'].unique()
    responses = {}
    for user_id in tqdm(unique_user_ids):
        if user_id not in responses:
            user_prompt = df[df['user_idx'] == user_id]['user_prompt'].iloc[0]
            responses[user_id] = generate_text(llm,tokenizer,user_prompt)
    df['new_user_prompt'] = df['user_idx'].map(responses)
    logging.info("New user prompt added successfully.")
    return df

def retrain_rec_model(args, train_set, test_set,train_loader,test_loader):
    model = load_rec_model(args, test_set)
    optimizer, scheduler = get_optimizer(args, model, len(train_set))

    save_path = os.path.join(args.save_dir, f'{args.algo}_trial.pt')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    best_auc = 0
    global_step = 0
    patience = 0
    for epoch in tqdm(range(args.epoch_num)):
        t = time.time()
        train_loss = []
        model.train()
        for _, data in enumerate(train_loader):
            outputs = model(data)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            global_step += 1
        train_time = time.time() - t
        # eval_auc, eval_ll, eval_loss, eval_time = model_eval(model, test_loader)
        # print("EPOCH %d  STEP %d train loss: %.5f, train time: %.5f, test loss: %.5f, test time: %.5f, auc: %.5f, "
        #       "logloss: %.5f" % (epoch, global_step, np.mean(train_loss), train_time, eval_loss,
        #                          eval_time, eval_auc, eval_ll))
        # if eval_auc > best_auc:
        #     best_auc = eval_auc
        #     torch.save(model, save_path)
        #     print('model save in', save_path)
        #     patience = 0
        # else:
        #     patience += 1
        #     if patience >= args.patience:
        #         break
    torch.save(model, save_path)
    return model

    
def get_training_args():
    args = parse_args()
    args.algo = 'DIN'
    args.save_dir = r'/home/yandex/DL20232024a/asafavrahamy/RefRec/RS/model'
    args.data_dir = '../data/ml-1m/proc_data'
    args.task_name = 'ctr'
    args.dataset_name = 'ml-1m'
    args.aug_prefix = 'bert_avg'
    args.augment = True
    args.epoch_num = 2
    args.batch_size = 32
    args.lr = 0.001
    args.lr_sched = 'cosine'
    args.weight_decay = 0  
    args.model = 'DIN'
    args.embed_size = 32
    args.final_mlp = [200,80]
    args.convert_arch = [128,32]
    args.num_cross_layers = 3
    args.dropout = 0.0  
    args.convert_type = 'HEA'
    args.convert_dropout = 0.0
    args.export_num = 2
    args.specific_export_num = 3
    args.dien_gru = 'AIGRU'
    args.device = 'cuda'
    return args