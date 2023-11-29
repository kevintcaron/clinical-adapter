import os
import spacy
import yaml
import pandas as pd
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import torch 
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
from utils import AssertionDatai2b2
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer # ,BertAdapterModel
from adapters import AdapterSetup, AutoAdapterModel

# from transformers import AutoTokenizer, AutoModel, AdapterTrainer, EvalPrediction# , AutoAdapterModel
import argparse
# from transformers.adapters import BnConfig,SeqBnConfig,DoubleSeqBnConfig # (new version)
# from transformers.adapters import PfeifferConfig,HoulsbyConfig

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
spacy.util.fix_random_seed(seed)

def _split_data(all_line_data_filtered_df,frac):
    all_line_data_filtered_df_frac = all_line_data_filtered_df.sample(frac=frac).copy()

    print("fraction of examples to used for training: {:.2f}".format(frac))
    print("number of examples after sampling: {:,}\n".format(all_line_data_filtered_df_frac.shape[0]))

    X = all_line_data_filtered_df_frac['new_line']
    y = all_line_data_filtered_df_frac['label']

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=seed, stratify=y_test_valid)

    print(f"X shape {X.shape} y shape : {y.shape}")
    print(f"X_train shape {X_train.shape} y_train shape : {y_train.shape}")
    print(f"X_val shape {X_valid.shape} y_val shape : {y_valid.shape}")
    print(f"X_test shape {X_test.shape} y_test shape : {y_test.shape}")

    return (X_train,y_train),(X_valid,y_valid),(X_test ,y_test)

def _create_datasets(train,valid,test):

    (X_train,y_train) = train
    (X_valid,y_valid)= valid
    (X_test ,y_test)= test
    print()
    print("Encoding Labels .....")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encode = np.asarray(encoder.transform(y_train))
    y_valid_encode = np.asarray(encoder.transform(y_valid))
    y_test_encode = np.asarray(encoder.transform(y_test))

    train_df = pd.DataFrame(X_train)
    valid_df = pd.DataFrame(X_valid)
    test_df = pd.DataFrame(X_test)

    train_df['label'] = y_train_encode.tolist()
    valid_df['label'] = y_valid_encode.tolist()
    test_df['label'] = y_test_encode.tolist()

    ds = DatasetDict ({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(valid_df),
    'test': Dataset.from_pandas(test_df)
    })

    return ds

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument("--help", "-h", action="help")

    return parser

def train(ds:Dataset,model:AutoModel,tokenizer:AutoTokenizer,adapter:bool,lr:float,epochs:int, 
                            output_dir:str, device:str, args):
    
    model = model.to(device)
    special_tokens_dict = {"additional_special_tokens": ["[entity]"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict,False)
    print()
    print("We have added", num_added_toks, "tokens")
    print()
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(example):
        return tokenizer(example["new_line"],   padding="max_length", truncation=True)
    
    # Takes ds dataset and tokenizes
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds = tokenized_ds.remove_columns(["new_line"])
    tokenized_ds = tokenized_ds.remove_columns(["__index_level_0__"])
    tokenized_ds.set_format("torch")

    # If fine-tuning head only, freeze the base model
    if args.finetune == 'head':
        for name, layer in model.base_model.named_parameters():
            layer.requires_grad = False
    elif args.finetune == 'full':
        pass
    else:
        raise ValueError("finetune argument must be either 'head' or 'full'")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"Number of trainable parameters: {num_params}")
    print()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate= lr,
        num_train_epochs= epochs,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save model checkpoints at the end of each epoch
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,  # Only keep the last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # report_to="wandb",
        push_to_hub=False,
    )
    
    if adapter:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=compute_metrics)
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=compute_metrics)

    try:
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model Fine-tuning Completed")
    except Exception as e:
        print(e)
        print("Model Fine-tuning Failed")


def main():

    parser = _setup_parser()
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    preprocessed_data_path = "Data/preprocessed-data"
    train_data_path = "Data/concept_assertion_relation_training_data"
    reference_test_data_path = "Data/reference_standard_for_test_data"
    test_data_path = "Data/test_data"
    task_name = 'ast-detection'

    ast_i2b2 = AssertionDatai2b2(preprocessed_data_path=preprocessed_data_path,
                                 train_data_path=train_data_path,
                                 reference_test_data_path=reference_test_data_path,
                                 test_data_path=test_data_path)
    
    beth_and_partners_ast, all_ast = ast_i2b2.load_assertion_i2b2_data()

    if args.i2b2 == 'all':
        train_data, valid_data, test_data = _split_data(all_ast,args.frac)
    elif args.i2b2 == 'beth_and_partners':
        train_data, valid_data, test_data = _split_data(beth_and_partners_ast,args.frac)
    else:
        raise ValueError("i2b2 argument must be either 'all' or 'beth_and_partners'")
    
    ds = _create_datasets(train_data,valid_data, test_data)

    if args.hd == 'arm':
        # FOR MAC
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print("device in use:", device)
    elif args.hd == 'intel':
        # FOR NVIDIA GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print()
        print("device in use:", device)
        print("GPU in use:", torch.cuda.get_device_name(0))
        print()

        # If using GPU, set seed for CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device('cpu')

    # Assign model and output directory
    if args.model == 'clinicalbert':
        pretrained_model_name_or_path = "emilyalsentzer/Bio_Discharge_Summary_BERT"
        output_dir= "clinicalbert_trainer"
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == "bert":
        pretrained_model_name_or_path = "bert-base-uncased"
        output_dir= "bert_trainer"
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError("model argument must be either 'clinicalbert' or 'bert'")

    # Assign use of adapter or not
    if args.adapter:

        # TODO: update to new adapters version
        tokenizer  = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, model_max_length=150)
        # model = BertAdapterModel.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path)
        model = AutoAdapterModel.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path)

        #PfeifferConfig,HoulsbyConfig
        if args.adapter_method == 'Pfeiffer':
            model.add_adapter(task_name,config=PfeifferConfig())
        else:
            model.add_adapter(task_name,config=HoulsbyConfig())
        model.train_adapter(task_name)
        model.add_classification_head(task_name, num_labels=3,id2label={0: 'PRESENT', 1: 'ABSENT', 2:'POSSIBLE'})
        model.set_active_adapters(task_name)

    else:
        tokenizer  = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, model_max_length=150)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, 
                                                                    num_labels=3,
                                                                    id2label={0: 'PRESENT', 1: 'ABSENT', 2:'POSSIBLE'})

    train(ds,model,tokenizer,args.adapter,args.lr,args.epochs,output_dir, device, args)


    # TODO
    # update to new adapters version
    # evaluate() - calculate f1 score
    # log experiments resutls - Table 1 ( in the paper)
    # Hyperparams tuning (lr) what else ??
    # save weights - maybe different output_dir for each experiments
    # load weights to test inference
    # load single tasks adapters
    # train adapters in parallel
    # train adapters fusion
    # data needed for plots Figure 3, Table 2, Figure 4





if __name__=="__main__":
    main()

    
    