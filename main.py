import logging
from config import Config, setup_logging
from data_utils import load_datasets, get_model_path, update_dataset_aug_vec, load_local_data, print_dataset_info,\
                        load_full_datasets, update_val_new_profile
from model_utils import load_model, evaluate_model, add_new_prompt_to_df, load_llm_unsloth, load_encoder_model, \
                        retrain_rec_model, get_training_args, reflect_and_improve_profile, extract_user_profile
import pandas as pd
import pickle

def main(load_df_llm = False, load_local_sets = False, save_small_sets = False, retrain = False):

    # Analyze small datasets
    if load_local_sets:
        logging.info("Loading local datasets")
        df_train, train_set, train_loader, df_val, val_set, val_loader = load_local_data(r'temp/small_train_set.pkl', r'temp/small_val_set.pkl')
    else:
        logging.info("Loading small datasets")
        df_train, train_set, train_loader, df_val, val_set, val_loader = load_full_datasets()
        # df_train, train_set, train_loader, df_val, val_set, val_loader = load_datasets(return_df=True, create_small_sets=True)
        
    if save_small_sets:
        logging.info("Saving small datasets")
        with open('small_train_set.pkl', 'wb') as f:
            pickle.dump(train_set, f)
        with open('small_val_set.pkl', 'wb') as f:
            pickle.dump(val_set, f)
        return 1
 
    
    print_dataset_info(df_train)
    print_dataset_info(df_val)
  
    # Load model
    logging.info("Loading model and encoder")
    model_path = get_model_path()
    rec_model = load_model(model_path)
    encoder_tokenizer, encoder_model = load_encoder_model()
    logging.info("Model and encoder loaded successfully")

    # Evaluate model
    logging.info("Evaluating initial model")
    df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds')
    logging.info(f"df_train predictions and labels: {df_train[['preds', 'labels']][:10].to_dict(orient='records')}")
    # Add new prompt to df_train
    logging.info("Loading LLM and tokenizer")
    llm, tokenizer = load_llm_unsloth()

    if load_df_llm:
        logging.info("Loading df_train with LLM")
        df_train = pd.read_pickle('temp/df_train_with_llm.pkl')
    else:
        logging.info("Adding new prompt to df_train")
        df_train = add_new_prompt_to_df(df_train, llm, tokenizer)

        # Save df_train with LLM
        df_train.to_pickle('df_train_with_llm.pkl')
        logging.info("df_train with LLM saved successfully.")

    #update df_val with new_prompt
    df_val = update_val_new_profile(df_train.copy(), df_val, 'new_user_prompt')

    # Train the model on the new data
    logging.info("Updating datasets with new user prompt")
    train_set, train_loader = update_dataset_aug_vec(df_train, 'new_user_prompt', train_set, encoder_model, encoder_tokenizer)
    val_set, val_loader = update_dataset_aug_vec(df_val, 'new_user_prompt', val_set, encoder_model, encoder_tokenizer)

    # Check preds before retraining
    logging.info("Evaluating model with new prompts before retraining")
    df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds_llama3')
    df_val = evaluate_model(rec_model, val_loader, df_val, column_name='preds_llama3')
    logging.info(f"df_train predictions and labels: {df_train[['preds_llama3', 'labels']][:10].to_dict(orient='records')}")
    logging.info(f"df_val predictions and labels: {df_val[['preds_llama3', 'labels']][:10].to_dict(orient='records')}")

    if retrain: 
        logging.info("Getting training arguments")
        args = get_training_args()
        logging.info(f"Args: {args}")
        # Retrain the model
        logging.info("Retraining the model")    
        rec_model = retrain_rec_model(args, new_train_set, val_set, train_loader, val_loader, trial_name='llama')
        logging.info("Evaluating retrained model")
        df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds_llama3_retrain')
        logging.info(f"df_train predictions and labels: {df_train[['preds_llama3_retrain', 'labels']][:10].to_dict(orient='records')}")

        # Perform self-reflection and improve the prompt
        logging.info("Performing self-reflection and improving prompt")
        # current_performance = df_train['preds_llama3_retrain'].mean()
    
    logging.info("Reflecting on user profile")
    df_train, improved_prompt = reflect_and_improve_profile(df_train, llm, tokenizer)
    df_train.to_pickle('df_train_with_improved_prompt.pkl')
    logging.info("df_train with improved prompt saved successfully.")
    
    user_id = df_train['user_idx'].unique()[0]
    logging.info(f"Improved prompt for user {user_id}: {improved_prompt[user_id]}")

    # Update the dataset with the improved prompt
    logging.info("Updating datasets with improved prompt")
    train_set, train_loader = update_dataset_aug_vec(df_train, 'improved_profile', train_set, encoder_model, encoder_tokenizer)
    val_set, val_loader = update_dataset_aug_vec(df_val, 'improved_profile', val_set, encoder_model, encoder_tokenizer)

    # Evaluate the model before retrain with the improved prompt
    logging.info("Evaluating model before retrain with improved profile")
    df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds_improved')
    df_val = evaluate_model(rec_model, val_loader, df_val, column_name='preds_improved')
    logging.info(f"df_train predictions and labels: {df_train[['preds_improved', 'labels']][:10].to_dict(orient='records')}")
    logging.info(f"df_val predictions and labels: {df_val[['preds_improved', 'labels']][:10].to_dict(orient='records')}")

    if retrain: 
        # Retrain the model with the improved prompt
        logging.info("Retraining model with improved prompt")
        rec_model = retrain_rec_model(args, new_train_set, val_set, train_loader, val_loader, trial_name='improved')

        # Evaluate the model with the improved prompt
        logging.info("Evaluating model after retrainwith improved profile")
        df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds_improved_retrain')
        df_val = evaluate_model(rec_model, val_loader, df_val, column_name='preds_improved_retrain')
        logging.info(f"df_train['preds_improved_retrain']: {df_train['preds_improved_retrain'][:10]}")

    # Compare the results
    logging.info("Comparing results")
    logging.info(f"Original performance: {df_train['preds_llama3'].mean()}")
    logging.info(f"Improved performance: {df_train['preds_improved'].mean()}")

    # Save dfs
    logging.info("Saving dataframes")
    df_train.to_csv('temp/df_train.csv', index=False)
    df_val.to_csv('temp/df_val.csv', index=False)
    logging.info("Dataframes saved successfully")


def df_val_evaluation():
    df_train, train_set, train_loader, df_val, val_set, val_loader = load_full_datasets()
    df_train = pd.read_csv('df_train_final.csv')
    df_val = pd.read_csv('df_val_final.csv')
    df_train['improved_profile'] = df_train['improved_profile'].apply(lambda x: extract_user_profile(x))
    df_val = update_val_new_profile(df_train.copy(), df_val, 'improved_profile')
    model_path = get_model_path()
    # model_path =r'/home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/RS/model/DIN_trial.pt'
    rec_model = load_model(model_path)
    df_val = evaluate_model(rec_model, val_loader, df_val, column_name='preds')

    encoder_tokenizer, encoder_model = load_encoder_model()
    val_set, val_loader = update_dataset_aug_vec(df_val, 'improved_profile', val_set, encoder_model, encoder_tokenizer)
    # df_train = evaluate_model(rec_model, train_loader, df_train, column_name='preds_llama3_improv')
    df_val = evaluate_model(rec_model, val_loader, df_val, column_name='preds_llama3_improv')
    
    df_train.to_csv('df_train_res.csv', index=False)
    df_val.to_csv('df_val_res.csv', index=False)
    return df_train, df_val


if __name__ == '__main__':
    setup_logging()
    logging.info("Starting application")
    main()
    # df_val_evaluation()
    logging.info("RefRec completed successfully.")
