# clinical-adapter
GA Tech CS7643 Group Project implementing adapter-transformers for clinical entity extraction and assertion classification tasks.

Overleaf Report (edit link): 
https://www.overleaf.com/2545733711ycgbpndtfgbf#fbe4c1

Example Overleaf (for refernce): 
https://www.overleaf.com/project/5f5ec061aa94370001943266





Project Summary:
Advancements in natural language processing (NLP) and natural language understanding (NLU) offer new and exciting applications to the fields of healthcare and public health. Specifically, extracting important pieces of information in various types of health records and assessing the certainty of clinical statements represents an important task with applications in the medical industry, public health, and several fields of research. However, currently, these domains face challenges related to a lack of resources and techniques to efficiently solve the disparate and complex tasks needed to evaluate health records. One solution is to use transfer learning, leveraging pre-trained models from the Bidirectional Encoder Representations from Transformers (BERT) family, and further fine-tuning them in the healthcare domain to develop specific task models. However, this still requires resources to fully fine-tune multiple models on specific tasks or subtasks. In recent years, several new approaches to transfer and multitask learning using "adapter transformers" have been proposed. These approaches serve as efficient parameter fine-tuning techniques, reducing the number of parameters and storage of models.
This project aims to explore approaches of parameter-efficient fine-tuning using adapters and evaluate their application in multitask learning on two linked NLU tasks using healthcare records: clinical entity extraction and clinical assertion classification.

## To run hyperparameter tuning on wandb

Change the hyperparameters that are defined un run_experiment.ipynb in the constant sweep_configuration. The current code only allows hyperparameter tuning of parameters that are included in the config file. While chaning hyperparameters do make sure that the key in the json matches with key in config.yaml and for hyperparameters that have int or float values.

