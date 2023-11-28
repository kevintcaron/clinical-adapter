
import os 
import re
import pandas as pd 
import spacy
import re
from sklearn.model_selection import train_test_split
import numpy as np

def load_clinical_notes(records_files,data_path):
    # reading the data files in a list
    content_records = []
    for record in records_files:
        _file = os.path.join(data_path,record)
        with open(_file) as f:
            content = f.read()
            #lines = content.split("\n")
            content_records.append((record[:-4],content))

    f.close()

    return content_records



def split_note_sentences(content_records):
    # load spacy
    nlp1 = spacy.load("en_ner_bc5cdr_md",disable = ['parser'])
    nlp1.add_pipe('sentencizer')

    # transform the data into a list of sentences
    docs = [(r,nlp1(text)) for r,text in content_records]
    data = []
    for r,doc in docs:
        for s in doc.sents:
            sentence = str.strip(str(s))
            sentence = sentence.replace("\n"," ")
            data.append((r,sentence))
    
    return data



def load_notes_labels(records,labels_path):

    records_files_ast = [f"record-{i}.ast" for i in records]
    
    # load labels in a list
    labels_records = []
    for record in records_files_ast:
        _file = os.path.join(labels_path,record)
        #print(_file)
        with open(_file) as f:
            content = f.readlines()
            file_data = []
            for line in content:
                ast = line.strip().split('||')
                line_entity = []

                assertion = ast[2].split('=')
                entity_label = ast[1].split("=")
        
                entity_text = re.findall('"([^"]*)"', ast[0])

                line_entity.append(record[:-4])
                line_entity.append(assertion[1].replace('"',''))
                line_entity.append(entity_label[1].replace('"',''))
                line_entity.append(entity_text[0])
                file_data.append(line_entity)
        labels_records.append(file_data)

    f.close()

    return labels_records



def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # remove all non-ASCII characters:
    text = re.sub(r'[^\x00-\x7f]',r'', text) 
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    text = " ".join(text.split())
    return text



def linking_train_labels_data(data,df_data_labels):
    new_data = []
    for r , sent in data:
        for index,row in df_data_labels.loc[df_data_labels['record'] == r,['entity','assertion']].iterrows():
            entity = clean_text(row['entity'])
            sentence = clean_text(sent)
            #print(entity,sent)
            try:
                if re.search(r'\b' + str(entity) + r'\b', str(sentence)):
                    new_data.append((r,entity,sentence,row['assertion']))
            except:
                print(r)
                print("entity:",str(entity))
                print("****")
    return new_data

def annotate_data(new_data):
    processed_data = []
    for r,entity,text,label in new_data:
        #print(text)
        match = re.search(r'\b' + entity + r'\b',text)

        res = list(text)
        res.insert(match.start(), '[entity] ')
        res.insert(match.end()+1, ' [entity]')
        res = ''.join(res)
        processed_data.append((r,entity,res,label))  

    return processed_data


def prepare_data(processed_data,frac=0.2,train_size=0.8,test_size=0.1):
    
    prepare_data = [{'sentence':text , 'label':label,'idx':idx} for idx,(r, entity, text,label) in enumerate(processed_data)]

    df_i2b2 = pd.DataFrame(prepare_data)
    df_i2b2 = df_i2b2[(df_i2b2.label=='present') | (df_i2b2.label=='absent') | (df_i2b2.label=='possible') ].copy()
    df_i2b2 = df_i2b2.sample(frac=frac).copy()

    X = df_i2b2['sentence']
    y = df_i2b2['label']

    X_train_valid,X_test,y_train_valid, y_test= train_test_split(X,y,test_size=test_size,stratify=y,random_state=42)
    X_train,X_valid,y_train,y_valid = train_test_split(X_train_valid,y_train_valid,train_size=train_size,random_state=42,stratify=y_train_valid)

    print(f"X_train shape {X_train.shape} y_train shape : {y_train.shape}")
    print(f"X_valid shape {X_valid.shape} y_valid shape : {y_valid.shape}")
    print(f"X_test shape {X_test.shape} y_test shape : {y_test.shape}")

    rs = {'train':np.vstack((y_train,X_train)),
          'validation':np.vstack((y_valid,X_valid)),
          'test':np.vstack((y_test,X_test))}

    return  rs

def load_dataset_assertion_classification(frac=0.2,train_size=0.8,test_size=0.1):
    cwd  = os.getcwd()
    labels_path = os.path.join(cwd,"Data/concept_assertion_relation_training_data","beth","ast")
    data_path = os.path.join(cwd,"Data/concept_assertion_relation_training_data","beth","txt")

    # creating a list of the files names
    records = [i for i in range(13, 39)]
    records = records + [i for i in range(45, 57)]
    records = records + [58,59]
    records = records + [i for i in range(65, 71)]
    records = records + [73,74]
    records = records + [i for i in range(81, 85)]
    records = records + [i for i in range(105,109)]
    records = records + [i for i in range(121,125)]
    records = records + [i for i in range(140,145)]
    records = records + [i for i in range(175,180)]
    records_files = [f"record-{i}.txt" for i in records]

    content_records = load_clinical_notes(records_files,data_path)

    data = split_note_sentences(content_records)
    labels_records  = load_notes_labels(records,labels_path)

    # labels in a dataframe
    data_labels = [line for f in labels_records for line in f]
    df_data_labels = pd.DataFrame(data_labels,columns=['record','assertion','label','entity'])
    new_data = linking_train_labels_data(data,df_data_labels)

    processed_data =  annotate_data(new_data)

    data_dict = prepare_data(processed_data,frac,train_size,test_size)

    return data_dict['train'], data_dict['validation'], data_dict['test']


if __name__ =="__main__":

    train, val, test = load_dataset_assertion_classification(frac=0.99,train_size=0.8,test_size=0.1)



    