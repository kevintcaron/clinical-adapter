import os
import re
import en_ner_bc5cdr_md
import spacy
import copy
import pandas as pd
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import torch 
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW


class AssertionDatai2b2():
    def __init__(self,preprocessed_data_path,train_data_path,reference_test_data_path,test_data_path) -> None:
        self.train_data_path = train_data_path
        self.reference_test_data_path = reference_test_data_path # test labels
        self.test_data_path = test_data_path
        self.preprocessed_data_path = preprocessed_data_path

    def load_clinical_notes(self):
        # Get paths for data and labels
        cwd  = os.getcwd()
        processed_path = os.path.join(cwd, f"{self.preprocessed_data_path}")
        labels_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","ast")
        data_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","txt")
        labels_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","ast")
        data_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","txt")
        labels_path_test = os.path.join(cwd,f"{self.reference_test_data_path}","ast")
        data_path_test = os.path.join(cwd,f"{self.test_data_path}")
        print('beth ast path', labels_path_beth)
        print('beth txt path', data_path_beth)
        print('partners ast path', labels_path_partners)
        print('partners txt path', data_path_partners)
        print('test ast path', labels_path_test)
        print('test txt path', data_path_test)

        return labels_path_beth,data_path_beth,labels_path_partners,data_path_partners,labels_path_test,data_path_test
        
    def clean_text(self,text):
        """
        Applies some pre-processing on the given text.
        Returns clean text
        
        Steps :
        - Removes HTML tags
        - Removes punctuation
        - lowers text
        """
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"\\", "", text)    
        text = re.sub(r"\'", "", text)    
        text = re.sub(r"\"", "", text)    
        text = text.strip().lower()
        text = re.sub(r'[^\x00-\x7f]',r'', text) 
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
        text = " ".join(text.split())
        
        return text

    def remove_adjacent_periods(self,text):
        """
        Takes a text string
        Returns the text string without adjacent periods (see 'Dr.' to 'Dr' below while keeping end period)
        
        For example:
        input: 'Admitted directly to OR from ambulance transfer and underwent cabg x3 with Dr. Howard on 06-13 .'
        output: 'Admitted directly to OR from ambulance transfer and underwent cabg x3 with Dr Howard on 06-13 .'

        This is needed for preventing the sentencizer from splitting on periods that are inside sentences.
        
        """
        pattern = re.compile(r'(?<=\S)\.')
        modified_text = re.sub(pattern, '', text)
        
        return modified_text

    def list_files_in_directory(self,directory_path):
        """
        Takes the path to the records.txt files
        Returns a list of the names of the records in each path
        """
        ignore = ['.DS_Store']
        files = os.listdir(directory_path)
        files = [file for file in files if os.path.isfile(os.path.join(directory_path, file)) and file not in ignore]
        files = [file[:-4] for file in files]
        
        return files

    def load_record_content_dict(self,notes_list, data_path):
        """
        Takes the list of names of the records and the path to the data records.txt files
        
        Returns a dictionary containing the record_name associated with all the text in the record.txt file:
        content_dict = {record: <all text in the record.txt file>}
        
        eg. 
        content_dict = {'record-105': 'Admission Date :\n2017-06-13\nDischarge Date :\n2017...( End of Report )'}
        
        """
        content_notes = {}
        for note in notes_list:
            _file = os.path.join(data_path, note + '.txt')
            with open(_file) as f:
                content = f.read()
                content_notes[note] = content
        return content_notes

    def load_record_labels_dict(self,notes_list, labels_path):
        """
        Takes the list of names of the records and the path to the data records.txt files
        
        Returns a dictionary containing the record_name and all the text in the record.ast file:
        labels_dict = {record: <all text in the record.ast file>}
        
        For example:
        labels_dict = {'record-105': 'c="left basilar atelectasis" 55:6 55:8||t="problem"||a="present"\nc="ventral ...'}
        
        """
        labels_notes = {}
        for note in notes_list:
            _file = os.path.join(labels_path, note + '.ast')
            with open(_file) as f:
                content = f.read()
                line_dict = {}
                for line in content:
                    
                    
                    labels_notes[note] = content
                
        f.close()
        return labels_notes

    def create_labels_dict(self,labels_dict):
        """
        Takes labels dictionary:
        labels_dict = {record: <all text in the record.ast file>}
        
        Returns a new dictionary of the labels for each record on each line of the record:
        new_labels_dict = {record: {ast_line: {'location': [<list of locations of entity in record.txt file>],
                                            'problem': '<the entity that is the target for assertion classification>',
                                            'assertion:' 'the assertion label'}}}

        For example:
        new_labels_dict = {'record-105': {1 : {'location': [55, 6, 55, 8],
                                            'problem': 'left basilar atelectasis',
                                            'assertion': 'present'}, 
                                        ...
                                        n :  {...}}
        """
        labels_dict_new = {}
        for key in labels_dict:
            lines = labels_dict[key].split("\n")
            line_dict = {}
            i = 1
            for line in lines:
                if line != '':
                    ast = line.strip().split('||')
                    assertion = re.findall('"([^"]*)"', ast[2])[0]
                    entity = re.split(r'" \d+', ast[0])[0][3:]
                    loc_section = re.split(r'" (\d)', ast[0])
                    loc = "".join([loc_section[1], loc_section[2]])
                    start, end = loc.split(' ')
                    s_line, s_tok = start.split(':')
                    e_line, e_tok = end.split(':')
                    line_info = {'location': [int(s_line), int(s_tok), int(e_line), int(e_tok)], 'problem': entity, 'assertion': assertion}
                    line_dict[i] = line_info
                    i+= 1
            labels_dict_new[key] = line_dict
                
        return labels_dict_new

    def create_lined_content_dict(self,content_dict):
        """
        Takes the content dictionary:
        content_dict = {record: <all text in the record.txt file>}

        Returns a new dictionary of the content line-by-line, with a key for each line
        new_content_dict = {'record': {1: <all the text on line 1 of record.txt>,
                                    2: <all the text on line 2 of record.txt>,
                                    ...,
                                    n: <all the text on last line (n) of record.txt>}}

        For example:
        new_content_dict = {'record-105': {1: 'Admission Date :',
                                        2: '2017-06-13',
                                        3: 'Discharge Date :',
                                        ...
                                        153: '( End of Report )'}}
        """
        new_content = {}
        for key in content_dict:
            lines = content_dict[key].split("\n")
            line_dict = {}
            i = 1
            for line in lines:
                line_dict[i] = line
                i += 1
                
            new_content[key] = line_dict
        return new_content

    def add_entity_and_assertion(self,new_labels, new_content):
        """
        Takes the new_content dictionary and the new_label dictionary:
        new_content_dict = {'record': {1: <all the text on line 1 of record.txt>,
                                2: <all the text on line 2 of record.txt>,
                                ...,
                                n: <all the text on last line (n) of record.txt>}}

        Returns a data dictionary that contains the record and each line of the record with infomration for the
        ast_line number, txt_line number, entity, old_line text, new_line text, and ast label.
        data = {'record': {1: {'ast_line': <line number the ast file (matches the key)>,
                            'txt_line': <line number of the txt file where the target entity is located>,
                            'entity': <the target entity (problem to be classified) from the line in the record.ast file>,
                            'old_line': <all the text on the line in record.txt where the target entity is located>,
                            'new_line': <all the text on the line in record.txt where the target entity is located 
                                        with [entity] tokens inserted around the target entity>,
                            'label': <the label of the entity in the text (present, absent, possible, etc.)>},
                            ...
                            n: {...}}

        For example:
        data = {'record-105': {1: {'ast_line': 1,
                                'txt_line': 55,
                                'entity': 'left basilar atelectasis',
                                'old_line': 'There has been interval improvement in left basilar atelectasis .',
                                'new_line': 'There has been interval improvement in [entity] left basilar atelectasis [entity] .',
                                'label': 'present'},
                            ...
                            n: {...}}
        """
        data = {}
        for record in new_labels:
            i = 1
            data[record] = {}
            for key in new_labels[record]:
                s_line = new_labels[record][key]['location'][0]
                s_tok = new_labels[record][key]['location'][1]
                e_line = new_labels[record][key]['location'][2]
                e_tok = new_labels[record][key]['location'][3]
                assertion = new_labels[record][key]['assertion']
                entity = new_labels[record][key]['problem']
                old_line = new_content[record][s_line]
                
    #             entity = clean_text(entity)
    #             old_line = clean_text(old_line)
                entity = self.remove_adjacent_periods(entity)
                old_line = self.remove_adjacent_periods(old_line)
                
                words = old_line.split()
                words.insert(s_tok, '[entity]')
                words.insert(e_tok + 2, '[entity]')
                new_line = ' '.join(words)
                
                data[record][i] = {'ast_line': i, 'txt_line': s_line, 
                                'entity': entity,'old_line': old_line, 
                                'new_line': new_line, 'label': assertion}
                i += 1
            
        return data

    def get_data_by_line(self,data):
        """
        Takes line_data dictionary
        Returns line data dataframe
        """
        record_dfs = []
        for record, info in data.items():
            df = pd.DataFrame.from_dict(info, orient='index')
            df['record'] = record
            record_dfs.append(df)
        final_df = pd.concat(record_dfs).reset_index(drop=True)
        final_df = final_df[['record', 'ast_line', 'txt_line', 'entity', 'new_line', 'label']]

        return final_df

    def filter_data(self,data_df):
        """
        Takes dataframe
        Returns filtered dataframe with only present, absent, possible and dropped null rows
        """
        allowed_labels = ['present', 'absent', 'possible']
        filtered_df = data_df[data_df['label'].isin(allowed_labels)]
        filtered_df = filtered_df.dropna()
        filtered_df.reset_index(inplace=True, drop=True)
        
        return filtered_df
    

    def load_assertion_i2b2_data(self):
        labels_path_beth,data_path_beth,labels_path_partners,data_path_partners,labels_path_test,data_path_test = self.load_clinical_notes()
        notes_beth = self.list_files_in_directory(data_path_beth)
        notes_partners = self.list_files_in_directory(data_path_partners)
        notes_test = self.list_files_in_directory(data_path_test)

        beth_content_dict = self.load_record_content_dict(notes_beth, data_path_beth)
        partners_content_dict = self.load_record_content_dict(notes_partners, data_path_partners)
        test_content_dict = self.load_record_content_dict(notes_test, data_path_test)

        # Merge the content dictionaries into one
        training_content_dict = {**beth_content_dict, **partners_content_dict}
        all_content_dict = {**training_content_dict, **test_content_dict}

        # Print the number of records for each dictionary
        print("number of beth training records:", len(beth_content_dict))
        print("number of partners training records:", len(partners_content_dict))
        print("number of all test records:", len(test_content_dict))
        print("number of combined beth and partners records:", len(training_content_dict))
        print("number of all combined records:", len(all_content_dict))


        beth_labels_dict  = self.load_record_labels_dict(notes_beth, labels_path_beth)
        partners_labels_dict = self.load_record_labels_dict(notes_partners, labels_path_partners)
        test_labels_dict = self.load_record_labels_dict(notes_test, labels_path_test)

        # # Merge the labels dictionaries into one
        training_labels_dict = {**beth_labels_dict, **partners_labels_dict}
        all_labels_dict = {**training_labels_dict, **test_labels_dict}

        # Print the number of records conatining labels for each dictionary
        print("number of beth records with labels:", len(beth_labels_dict))
        print("number of partners records with labels:", len(partners_labels_dict))
        print("number of test records with labels:", len(test_labels_dict))
        print("number of combined beth and partners records with labels:", len(training_labels_dict))
        print("number of all combined records with labels:", len(all_labels_dict))

        training_labels = self.create_labels_dict(training_labels_dict)
        test_labels = self.create_labels_dict(test_labels_dict)
        all_labels = self.create_labels_dict(all_labels_dict)
        
        training_content = self.create_lined_content_dict(training_content_dict)
        test_content = self.create_lined_content_dict(test_content_dict)
        all_content = self.create_lined_content_dict(all_content_dict)

        training_line_data = self.add_entity_and_assertion(training_labels, training_content)
        test_line_data = self.add_entity_and_assertion(test_labels, test_content)
        all_line_data = self.add_entity_and_assertion(all_labels, all_content)

        training_line_data_df = self.get_data_by_line(training_line_data)
        test_line_data_df = self.get_data_by_line(test_line_data)
        all_line_data_df = self.get_data_by_line(all_line_data)

        training_line_data_filtered_df = self.filter_data(training_line_data_df)
        test_line_data_filtered_df = self.filter_data(test_line_data_df)
        all_line_data_filtered_df = self.filter_data(all_line_data_df)

        return training_line_data_filtered_df,test_line_data_filtered_df,all_line_data_filtered_df
