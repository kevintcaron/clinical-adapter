import os
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
from utils import AssertionDatai2b2, ConceptDatai2b2
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForTokenClassification# ,BertAdapterModel
from adapters import AdapterSetup, AutoAdapterModel,AdapterTrainer
from adapters import SeqBnConfig,DoubleSeqBnConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import ast
import evaluate


# from transformers import AutoTokenizer, AutoModel, AdapterTrainer, EvalPrediction# , AutoAdapterModel
import argparse
# from transformers.adapters import BnConfig,SeqBnConfig,DoubleSeqBnConfig # (new version)
# from transformers.adapters import PfeifferConfig,HoulsbyConfig

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# spacy.util.fix_random_seed(seed)

def _split_data(all_line_data_filtered_df,frac, test_data=None, i2b2='all',task_name='ast'):
    all_line_data_filtered_df_frac = all_line_data_filtered_df.sample(frac=frac).copy()

    print("fraction of examples to used for training: {:.2f}".format(frac))
    print("number of examples after sampling: {:,}\n".format(all_line_data_filtered_df_frac.shape[0]))
    # print(all_line_data_filtered_df)
    if task_name == 'ast':
      X = all_line_data_filtered_df_frac['new_line']
      y = all_line_data_filtered_df_frac['label']
      X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
      X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=seed, stratify=y_test_valid)
    elif task_name == 'ner':
      X = all_line_data_filtered_df_frac['tokens']
      y = all_line_data_filtered_df_frac['ner_tags']
      X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=seed)
      if i2b2 == 'beth_and_partners':
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
      elif i2b2 == 'all':
        X_test = test_data['tokens']
        y_test = test_data['ner_tags']
    

    print(f"X shape {X.shape} y shape : {y.shape}")
    print(f"X_train shape {X_train.shape} y_train shape : {y_train.shape}")
    print(f"X_val shape {X_valid.shape} y_val shape : {y_valid.shape}")
    print(f"X_test shape {X_test.shape} y_test shape : {y_test.shape}")

    return (X_train,y_train),(X_valid,y_valid),(X_test ,y_test)

def _create_datasets(train,valid,test, task='ast'):

    (X_train,y_train) = train
    (X_valid,y_valid)= valid
    (X_test ,y_test)= test
    print()
    if task == 'ast':
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
        
    elif task == 'ner':
        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()

        train_df['tokens'] = [ast.literal_eval(t) for t in X_train]
        valid_df['tokens'] = [ast.literal_eval(t) for t in X_valid]
        test_df['tokens'] = [ast.literal_eval(t) for t in X_test]

        train_df['ner_tags'] = [ast.literal_eval(t) for t in y_train]
        valid_df['ner_tags'] = [ast.literal_eval(t) for t in y_valid]
        test_df['ner_tags'] = [ast.literal_eval(t) for t in y_test]

    else:
        raise ValueError("task argument in _create_datasets function should be 'ast' or 'ner'. By Default it is set to 'ast'")

    ds = DatasetDict ({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(valid_df),
        'test': Dataset.from_pandas(test_df)
        })
    

    return ds

# def compute_metrics(eval_pred):
#     metric = evaluate.load("accuracy")
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred, task_name):
    # print(eval_pred)
    logits, labels = eval_pred
    if task_name == 'ast':
      predictions = np.argmax(logits, axis=-1)
      accuracy = accuracy_score(labels, predictions)
      precision = precision_score(labels, predictions, average='weighted')
      recall = recall_score(labels, predictions, average='weighted')
      f1 = f1_score(labels, predictions, average='weighted')

      return {
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
          'f1': f1,
      }
    elif task_name == 'ner':
      seqeval = evaluate.load("seqeval")
      predictions = np.argmax(logits, axis=2)
      label_list  = ["O", "B-test", "I-test", "B-problem", "I-problem", "B-treatment", "I-treatment"]
      true_predictions = [
          [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]
      true_labels = [
          [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]

      results = seqeval.compute(predictions=true_predictions, references=true_labels)
      return {
          "accuracy": results["overall_accuracy"],
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"]
      }
    

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument("--help", "-h", action="help")

    return parser

def train(tokenized_ds:Dataset,model:AutoModel,tokenizer:AutoTokenizer,adapter:bool,lr:float,epochs:int,weight_decay:float,batch:int,logging_steps:int,
                            output_dir:str, device:str, task_name:str,args):
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"Number of trainable parameters: {num_params}")
    print()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate= lr,
        num_train_epochs= epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save model checkpoints at the end of each epoch 
        logging_dir="./logs", 
        logging_steps=logging_steps,
        save_total_limit=2,  # Only keep the last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb" if args.wandb else "none", # ee5f5f4d8ea5f77b94eddbf412a4426a08b9451c
        push_to_hub=False,
    )

    
    if adapter:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=lambda p: compute_metrics(p, task_name=task_name))
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=lambda p: compute_metrics(p, task_name=task_name))

    try:
        trainer.train()
        if adapter:
            model.save_adapter(f"adapter_{task_name}", task_name,with_head=True)
        else:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        print("Model Fine-tuning Completed")
    except Exception as e:
        print(e)
        print("Model Fine-tuning Failed")

def tokenize_and_align_labels(examples,tokenizer):
    examples_tokens = examples["tokens"]
    tokenized_inputs = tokenizer(examples_tokens, padding="max_length",truncation=True, is_split_into_words=True)

    labels = []
    examples_ner= examples[f"ner_tags"]
    for i, label in enumerate(examples_ner):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():

    parser = _setup_parser()
    args, overriding_args = parser.parse_known_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #overwriting config with args from commandline for hyperparameter tuning
    for item in overriding_args:
        key, value = item.split('=')
        key = key.lstrip('-')
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        setattr(args, key, value)
    # Print all arguments
    print()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()

    preprocessed_data_path = "Data/preprocessed-data"
    train_data_path = "Data/concept_assertion_relation_training_data"
    reference_test_data_path = "Data/reference_standard_for_test_data"
    test_data_path = "Data/test_data"
    task_name = args.task

    if task_name == 'ast':
        ast_i2b2 = AssertionDatai2b2(preprocessed_data_path=preprocessed_data_path,
                                 train_data_path=train_data_path,
                                 reference_test_data_path=reference_test_data_path,
                                 test_data_path=test_data_path)
        beth_and_partners_data, all_data = ast_i2b2.load_assertion_i2b2_data()
        id2label = {0: 'PRESENT', 1: 'ABSENT', 2:'POSSIBLE'}

    elif task_name == 'ner':
        ner_i2b2 = ConceptDatai2b2(preprocessed_data_path=preprocessed_data_path,
                                 train_data_path=train_data_path,
                                 reference_test_data_path=reference_test_data_path,
                                 test_data_path=test_data_path)
        beth_and_partners_data, test_data = ner_i2b2.load_concept_i2b2_data()    
        id2label = {0: "O",1: "B-test",2: "I-test",3: "B-problem",4: "I-problem",5: "B-treatment",6: "I-treatment"}
        label2id = {"O": 0,"B-test": 1,"I-test": 2,"B-problem": 3,"I-problem": 4,"B-treatment": 5,"I-treatment": 6}

        
    else:
        raise ValueError("task argument must be either 'ast' or 'ner'")
    
    if args.i2b2 == 'all':
        train_data, valid_data, test_data = _split_data(all_data,args.frac,test_data=test_data, i2b2='all', task_name=task_name)
    elif args.i2b2 == 'beth_and_partners':
        train_data, valid_data, test_data = _split_data(beth_and_partners_data,args.frac, test_data=None, i2b2='beth_and_partners', task_name=task_name)
    else:
        raise ValueError("i2b2 argument must be either 'all' or 'beth_and_partners'")
    
    ds = _create_datasets(train_data,valid_data, test_data,task_name)

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
        pretrained_model_name_or_path = "emilyalsentzer/Bio_Discharge_Summary_BERT"   #"emilyalsentzer/Bio_ClinicalBERT"  TBD
        output_dir= "clinicalbert_trainer/" + str(task_name)
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == "bert":
        pretrained_model_name_or_path = "bert-base-uncased"
        output_dir= "bert_trainer/" + str(task_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError("model argument must be either 'clinicalbert' or 'bert'")

    tokenizer  = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, model_max_length=150)

    def tokenize_function_ast(example):
        return tokenizer(example["new_line"],   padding="max_length", truncation=True)
    
    # Takes ds dataset and tokenizes
    if task_name == 'ast':
        special_tokens_dict = {"additional_special_tokens": ["[entity]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict,False)
        print()
        print("We have added", num_added_toks, "tokens")
        print()
        tokenized_ds = ds.map(tokenize_function_ast, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.remove_columns(["new_line"])
        tokenized_ds = tokenized_ds.remove_columns(["__index_level_0__"])
        
    if task_name == 'ner':
        tokenized_ds = ds.map(lambda example: tokenize_and_align_labels(example, tokenizer), batched=True)

    tokenized_ds.set_format("torch")
    # Assign use of adapter or not
    if args.adapter:
        # model = BertAdapterModel.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path)
        model = AutoAdapterModel.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path)
        
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        #PfeifferConfig,HoulsbyConfig
        if args.adapter_method == 'SeqBnConfig':
            model.add_adapter(task_name,config=SeqBnConfig(reduction_factor=args.reduction_factor))
        else:
            model.add_adapter(task_name,config=DoubleSeqBnConfig(reduction_factor=args.reduction_factor))
        model.train_adapter(task_name)
        if task_name == 'ast':
            model.add_classification_head(task_name, num_labels=len(id2label),id2label=id2label)
        elif task_name == 'ner':
            model.add_tagging_head("ner_head", num_labels=len(id2label), id2label=id2label)
        model.set_active_adapters(task_name)

    else:
        #tokenizer  = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, model_max_length=150)
        if task_name == 'ast':
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, 
                                                                        num_labels=len(id2label),
                                                                        id2label=id2label)
        elif task_name == 'ner':
            model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path,
                                                                    num_labels=len(id2label), 
                                                                    id2label=id2label, 
                                                                    label2id=label2id)     
        else:
            raise ValueError("task argument must be either 'ast' or 'ner'")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
    
        # If fine-tuning head only, freeze the base model
        if args.finetune == 'head':
            for name, layer in model.base_model.named_parameters():
                layer.requires_grad = False
        elif args.finetune == 'full':
            pass
        else:
            raise ValueError("finetune argument must be either 'head' or 'full'")

    # Set up wandb if True
    if args.wandb:
        # Set up wandb
        os.environ["WANDB_PROJECT"] = "clinical-adapter"  
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        is_type = "adapter" if args.adapter else "finetune"
        timestamp = int(time.time())

        # Create wandb run name
        if args.adapter:
            wandb_run_name = f"{task_name}-{args.model}-{args.i2b2}-{is_type}-{args.adapter_method}-{timestamp}"
        else:
            wandb_run_name = f"{task_name}-{args.model}-{args.i2b2}-{is_type}-{args.finetune}-{timestamp}"
        os.environ["WANDB_RUN_NAME"] = f"{wandb_run_name}"

    train(tokenized_ds,model,tokenizer,args.adapter,args.lr,args.epochs,args.weight_decay,args.batch,args.logging_steps,output_dir, device, task_name, args)


    # TODO
    # update to new adapters version
    # evaluate() - calculate f1 score
    # log experiments resutls - Table 1 ( in the paper)
    # Hyperparams tuning (lr) what else ??  added batch size, weight decay
    # save weights - maybe different output_dir for each experiments, add one for each task
    # load weights to test inference
    # load single tasks adapters
    # train adapters in parallel
    # train adapters fusion
    # data needed for plots Figure 3, Table 2, Figure 4





if __name__=="__main__":
    main()

    
    