
import logging

from config import Config, setup_logging
from data_utils import load_datasets, return_user_details, get_user_hist, get_item_desc, get_model_path, update_dataset_aug_vec
from model_utils import load_model, evaluate_model, add_new_prompt_to_df, load_llm, load_llm_unsloth, load_encoder_model, retrain_rec_model, get_training_args
import pandas as pd

def main():
    # Set up logging
    setup_logging()
    logging.info("Starting application")
    
    # Load datasets
    df_train, train_set, train_loader, df_test, test_set, test_loader = load_datasets()
    logging.info(f"Length of train dataset: {len(train_set)}")
    logging.info(f"Length of test dataset: {len(test_set)}")
    logging.info("Datasets loaded successfully.")
    df_train.to_csv('df_train_all.csv', index=False)
    return

    # Load model
    model_path = get_model_path()
    rec_model = load_model(model_path)
    encoder_tokenizer, encoder_model = load_encoder_model()

    # Evaluate model (example, you need to define the data_loader)
    df_test = evaluate_model(rec_model, test_loader, df_test, column_name='preds')
    logging.info(f"df_test['preds']: {df_test['preds'][:10]}")
    
    # Add new prompt to df_train
    llm, tokenizer = load_llm_unsloth()
    # df_train = add_new_prompt_to_df(df_train, llm, tokenizer)
    df_test = add_new_prompt_to_df(df_test, llm, tokenizer)
    df_train = pd.read_csv('df_train.csv')

    #train the model on the new data
    new_train_set, new_train_loader = update_dataset_aug_vec(df_train, 'new_user_prompt', train_set, encoder_model, encoder_tokenizer)
    new_test_set, new_test_loader = update_dataset_aug_vec(df_test, 'new_user_prompt', test_set, encoder_model, encoder_tokenizer)


    #check preds before retraining
    df_test = evaluate_model(rec_model, new_test_loader, df_test, column_name='preds_llama3')
    logging.info(f"df_test['preds_llama3']: {df_test['preds_llama3'][:10]}")


    args = get_training_args()
    #print args
    logging.info(f"Args: {args}")
    
    #retrain the model
    new_rec_model = retrain_rec_model(args, new_train_set, test_set, train_loader, test_loader)
    df_test = evaluate_model(new_rec_model, test_loader, df_test, column_name='preds_llama3_retrain')
    logging.info(f"df_test['preds_llama3_retrain']: {df_test['preds_llama3_retrain'][:10]}")

    #reflect on the results from the new model
    

    #save dfs
    df_train.to_csv('df_train.csv', index=False)
    df_test.to_csv('df_test.csv', index=False)

if __name__ == '__main__':
    main()
    logging.info("RefRec completed successfully.")

