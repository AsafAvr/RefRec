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
import re
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
    
    # Set pad token ID explicitly
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logging.info("LLM loaded successfully.")
    return model, tokenizer
    
def generate_text(model,tokenizer,prompt,max_seq_length = 2500):
    # Add padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=max_seq_length,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def llm_prompt(prompt_text,llm):
    system = "You are a helpful assistant in the movie recommendation domain. Please recommend without repeating the following text"
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm
    initial = chain.invoke({"text": prompt_text}).content
    return initial

def generate_reflection_prompt_old(user_profile, predictions, actual_ratings):
    reflection_prompt = f"""
    As an AI assistant specializing in movie recommendations, 
    your task is to analyze and improve the following user profile:

    Current User Profile:
    "{user_profile}"

    Recent Recommendations and Their Accuracy:
    {', '.join([f"{movie}: Predicted {pred:.2f}, Actual {actual}" for movie, pred, actual in zip(predictions['movie'], predictions['prediction'], actual_ratings)])}

    Based on this information, please reflect on the following:
    1. How well does the current user profile capture the user's preferences?
    2. What aspects of the user's taste might be missing or underrepresented in the profile?
    3. Are there any patterns in the prediction mismatches that suggest areas for improvement?
    4. How can we modify the user profile to improve future recommendations?

    Please provide an improved version of the user profile that addresses these points. 
    Focus on adjusting the profile to better match the user's actual ratings and preferences.
    Analyze user\'s preferences on movies (consider factors like genre, director/actors, time period/country, character, plot/theme, mood/tone, critical acclaim/award, 
    production quality, and soundtrack). Provide clear explanations based on relevant details from the user\'s movie viewing history and other pertinent factors.
    Your response should be in the form of a revised user profile, not an analysis.

    """
    return reflection_prompt


def generate_reflection_prompt(user_profile, predictions, actual_ratings):
    reflection_prompt = f"""
    As an AI assistant specializing in movie recommendations, your task is to analyze and improve the following user profile:

    Current User Profile:
    "{user_profile}"

    Recent Recommendations and Their Accuracy:
    {', '.join([f"{movie}: Predicted {pred:.2f}, Actual {actual}" for movie, pred, actual in zip(predictions['movie'], predictions['prediction'], actual_ratings)])}

    Based on this information, please perform the following tasks:

    1. Analyze Discrepancies:
       - Identify patterns in prediction errors (overestimations and underestimations).
       - Determine which genres, themes, or attributes are consistently misjudged.

    2. User Preference Analysis:
       - Infer deeper preferences based on highly rated movies.
       - Identify potential niche interests or unexpected favorites.

    3. Temporal Trends:
       - Consider if the user's tastes have evolved over time.
       - Assess if newer movies are rated differently from older ones.

    4. Contextual Factors:
       - Consider external factors that might influence ratings (e.g., directors, actors, release year).
       - Analyze if there are any mood-based or situational preferences.

    5. Profile Refinement:
       - Suggest specific additions or modifications to the user profile.
       - Propose new categories or attributes to better capture the user's taste.

    6. Confidence Levels:
       - Assign confidence levels to different aspects of the user profile.
       - Highlight areas where more data might be needed for accurate predictions.

    Based on this analysis, provide an improved version of the user profile that addresses these points. 
    The new profile should be more nuanced, capturing subtle preferences and potential exceptions.
    Format the new profile as a structured list of preferences, each with a confidence level (High, Medium, Low).

    Example format for the improved profile:
    - Genre Preferences:
      * Action (High Confidence): Enjoys high-paced, adrenaline-fueled movies, especially those with complex plots.
      * Romance (Medium Confidence): Appreciates romantic subplots in other genres, but rarely watches pure romance films.
    - Director Preferences:
      * Christopher Nolan (High Confidence): Consistently rates his films highly, particularly for their complex narratives.
    - Thematic Preferences:
      * Time Travel (High Confidence): Shows a strong interest in movies involving time manipulation or alternate timelines.
    - Actor Preferences:
      * Tom Hanks (Medium Confidence): Tends to rate his dramatic roles higher than his comedic ones.
    - Era Preferences:
      * 1990s Films (Low Confidence): Slight preference for movies from this decade, but more data needed.

    Ensure the improved profile is comprehensive, addressing all relevant aspects of movie preferences.
    """
    return reflection_prompt

def extract_user_profile(improved_profile):
    # Find the start of the actual profile
    start_marker = 'Revised User Profile:'
    end_marker = 'Revised User Profile Analysis:'
    
    start_index = improved_profile.find(start_marker)
    if start_index == -1:
        return improved_profile
    
    start_index += len(start_marker)
    
    end_index = improved_profile.find(end_marker)
    if end_index == -1:
        end_index = len(improved_profile)
    
    # Extract the profile
    profile = improved_profile[start_index:end_index].strip()
    
    # Remove any leading/trailing quotation marks and whitespace
    profile = re.sub(r'^["\s]+|["\s]+$', '', profile)
    
    return profile

def reflect_and_improve_profile(df_train, llm, tokenizer):
    improved_profiles = {}
    
    for user_id in tqdm(df_train['user_idx'].unique(), desc="Reflecting on user profiles"):
        user_data = df_train[df_train['user_idx'] == user_id]
        user_profile = user_data['user_hist'].iloc[0]  # Assuming 'user_profile' column exists
        
        # Get recent predictions and actual ratings for this user
        recent_preds = user_data[['item_desc', 'preds_llama3', 'labels']].tail(10)
        
        # Extract movie titles from item_desc
        recent_preds['movie_title'] = recent_preds['item_desc'].apply(lambda x: re.split(r'\s+is\s+', x, maxsplit=1)[0])
        
        reflection_prompt = generate_reflection_prompt(
            user_profile, 
            {'movie': recent_preds['movie_title'], 'prediction': recent_preds['preds_llama3']},
            recent_preds['labels']
        )
        
        improved_profile = generate_text(llm, tokenizer, reflection_prompt)
        improved_profile = extract_user_profile(improved_profile)
        improved_profiles[user_id] = improved_profile
    
    df_train['improved_profile'] = df_train['user_idx'].map(improved_profiles)
    return df_train, improved_profiles

def add_new_prompt_to_df(df, llm, tokenizer):
    unique_user_ids = df['user_idx'].unique()
    responses = {}
    for user_id in tqdm(unique_user_ids, desc="Adding new prompt to df"):
        if user_id not in responses:
            user_prompt = df[df['user_idx'] == user_id]['user_prompt'].iloc[0]
            responses[user_id] = generate_text(llm,tokenizer,user_prompt)
    df['new_user_prompt'] = df['user_idx'].map(responses)
    logging.info("New user prompt added successfully.")
    return df

def retrain_rec_model(args, train_set, test_set,train_loader,test_loader,trial_name = 'llama'):
    model = load_rec_model(args, test_set)
    optimizer, scheduler = get_optimizer(args, model, len(train_set))

    save_path = os.path.join(args.save_dir, f'{args.algo}_trial_{trial_name}.pt')
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