import os
import re
import pandas as pd
import ast
class AssertionDatai2b2():
    def __init__(self,preprocessed_data_path,train_data_path,reference_test_data_path,test_data_path) -> None:
        self.train_data_path = train_data_path
        self.reference_test_data_path = reference_test_data_path # test labels
        self.test_data_path = test_data_path
        self.preprocessed_data_path = preprocessed_data_path

    def load_clinical_notes(self):
        # Get paths for data and labels
        print()
        print("Loading data...")
        cwd  = os.getcwd()
        labels_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","ast")
        data_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","txt")
        labels_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","ast")
        data_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","txt")
        labels_path_test = os.path.join(cwd,f"{self.reference_test_data_path}","ast")
        data_path_test = os.path.join(cwd,f"{self.test_data_path}")

        return labels_path_beth,data_path_beth,labels_path_partners,data_path_partners,labels_path_test,data_path_test

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
                if content != '':
                    labels_notes[note] = content
                else:
                    print('no labels for', note)
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

                if s_line != e_line:
                    print('ERROR: entity spans multiple lines. Check record:')
                    print(record, key)
                    print(new_labels[record][key])
                    print(new_content[record][s_line])
                    print(new_content[record][e_line])
                    print()
                else:
                    pass

                assertion = new_labels[record][key]['assertion']
                entity = new_labels[record][key]['problem']
                old_line = new_content[record][s_line]
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
        print()
        print("number of beth training records:", len(beth_content_dict))
        print("number of partners training records:", len(partners_content_dict))
        print("number of all test records:", len(test_content_dict))
        print("total number of all combined records:", len(all_content_dict))
        print()

        beth_labels_dict  = self.load_record_labels_dict(notes_beth, labels_path_beth)
        partners_labels_dict = self.load_record_labels_dict(notes_partners, labels_path_partners)
        test_labels_dict = self.load_record_labels_dict(notes_test, labels_path_test)

        # Merge the labels dictionaries into one
        training_labels_dict = {**beth_labels_dict, **partners_labels_dict}
        all_labels_dict = {**training_labels_dict, **test_labels_dict}

        # Print the number of records containing labels for each dictionary
        print()
        print("number of beth records with labels:", len(beth_labels_dict))
        print("number of partners records with labels:", len(partners_labels_dict))
        print("number of test records with labels:", len(test_labels_dict))
        print("total number of all combined records with labels:", len(all_labels_dict))

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

        # Write the dataframes to csv
        cwd  = os.getcwd()
        path = os.path.join(cwd, f"{self.preprocessed_data_path}", "ast_line_data.csv")
        training_line_data_filtered_df.to_csv(path, index=False)

        # print the number of examples in each dataframe
        print()
        print("number of beth_and_partners examples:", len(training_line_data_filtered_df))
        print("number of test examples:", len(test_line_data_filtered_df))
        print("total number of combined examples:", len(all_line_data_filtered_df))
        print()

        return training_line_data_filtered_df, all_line_data_filtered_df

    def load_assertion_i2b2_data_from_file(self):
        cwd  = os.getcwd()
        path = os.path.join(cwd, f"{self.preprocessed_data_path}", "ast_line_data.csv")
        all_line_data_filtered_df = pd.read_csv(path)
        return all_line_data_filtered_df

class ConceptDatai2b2():
    def __init__(self,preprocessed_data_path,train_data_path,reference_test_data_path,test_data_path) -> None:
        self.train_data_path = train_data_path
        self.reference_test_data_path = reference_test_data_path # test labels
        self.test_data_path = test_data_path
        self.preprocessed_data_path = preprocessed_data_path

    def load_clinical_notes(self):
        # Get paths for data and labels
        print()
        print("Loading data for NER...")
        cwd  = os.getcwd()
        labels_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","ast")
        data_path_beth = os.path.join(cwd,f"{self.train_data_path}","beth","txt")
        labels_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","ast")
        data_path_partners = os.path.join(cwd,f"{self.train_data_path}","partners","txt")
        labels_path_test = os.path.join(cwd,f"{self.reference_test_data_path}","ast")
        data_path_test = os.path.join(cwd,f"{self.test_data_path}")

        return labels_path_beth,data_path_beth,labels_path_partners,data_path_partners,labels_path_test,data_path_test
    
    def computeConceptDict(self,conFilePath):
        cf = open(conFilePath,"r")
        cf_Lines = cf.readlines()
        line_dict = dict()

        for cf_line in cf_Lines:
            # print cf_line
            #c="a workup" 27:2 27:3||t="test"
            concept= cf_line.split("||")

            iob_wordIdx = concept[0].split()
            # print concept[0]
            iob_class = concept[1].split("=")
            iob_class = iob_class[1].replace("\"","")
            iob_class = iob_class.replace("\n","")

            # print iob_wordIdx[len(iob_wordIdx)-2],iob_wordIdx[len(iob_wordIdx)-1]
            start_iobLineNo = iob_wordIdx[len(iob_wordIdx)-2].split(":")
            end_iobLineNo = iob_wordIdx[len(iob_wordIdx)-1].split(":")
            start_idx = start_iobLineNo[1]
            end_idx = end_iobLineNo[1]
            iobLineNo=start_iobLineNo[0]

            if iobLineNo in line_dict.keys():
                    line_dict[iobLineNo].append(start_idx+"-"+end_idx+"-"+iob_class)
            else:
                    line_dict.update({iobLineNo:[start_idx+"-"+end_idx+"-"+iob_class]})
        return line_dict

    def prepareIOB_wordList(self,lineNumber,IOBwordList,conceptDict):

        iobTagList= []
        if str(lineNumber) in conceptDict.keys():
            # print conceptDict[str(lineNumber)]

            # split the tag and get the index of word and tag
            for concept in conceptDict[str(lineNumber)]:
                concept = str(concept).split("-")
                # print "start_idx, end_idx",concept[0],concept[1]
                # if (start_idx - end_idx) is zero then only B- prefix is applicable
                getrange = list(range(int(concept[0]),int(concept[1])))
                getrange.append(int(concept[1]))
                # For all the idx not in getrange assign an O tag
                # print getrange
                if(len(getrange) > 1):

                        for idx in range(0,len(getrange)):
                            # print getrange[idx]
                            iobTagList.append(int(getrange[idx]))
                            if(idx == 0):
                                    IOBwordList[getrange[idx]] = "B-"+concept[2]
                            else:
                                    IOBwordList[getrange[idx]] = "I-"+concept[2]
                else:
                        idx = getrange[0]
                        iobTagList.append(int(getrange[0]))
                        # print idx
                        IOBwordList[idx] = "B-"+concept[2]
                # Else for all the indices between start and end apply the I- prefix

                # For all the other words assign O tag
            for i in range(0,len(IOBwordList)):
                if i not in iobTagList:
                    IOBwordList[i] = "O"
            # print "IOB- WordList ",IOBwordList
        else:
            # print ""
            for i in range(0,len(IOBwordList)):
                if i not in iobTagList:
                    IOBwordList[i] = "O"
        return IOBwordList
    
    def handle_imbalanced_dataset(df, drop=0):
        """
        This function expects a dataframe with ner_tags as one of it's columns, which is pre-processed i2b2 concept data.
        It returns a dataframe, and also prints some statistics to check in the given dataset.
        if drop=1, then 80% of sentences with all others values will be dropped from the final dataframe randomly.
        """
        count_zero = 0
        count_non_zero = 0
        for index, row in df.iterrows():
            ner_tags = ast.literal_eval(row['ner_tags'])
            # Check if all elements in ner_tags are 0
            if all(tag == 0 for tag in ner_tags):
                count_zero += 1
            else:
                count_non_zero += 1
        # Calculate the ratio
        total_rows = len(df)
        ratio_zero = count_zero / total_rows
        ratio_non_zero = count_non_zero / total_rows
        print(f"Rows with all 0s in ner_tags: {count_zero}")
        print(f"Rows with non-zero values in ner_tags: {count_non_zero}")
        print(f"Ratio of rows with all 0s: {ratio_zero:.2%}")
        print(f"Ratio of rows with non-zero values: {ratio_non_zero:.2%}")
        
        if drop==1:
            df_to_remove = df[df['ner_tags'].apply(lambda tags: all(tag == 0 for tag in ast.literal_eval(tags)))].sample(frac=0.8)
            df = df.drop(df_to_remove.index)

        return df

    def create_concept_i2b2_data(self):
        # Define the mapping dictionary
        mapping_dict = {
            'O': 0,
            'B-test': 1,
            'I-test': 2,
            'B-problem': 3,
            'I-problem': 4,
            'B-treatment': 5,
            'I-treatment': 6
        }

        # # Example list of values
        # values_list = ['O', 'B-test', 'I-test', 'B-problem', 'I-problem', 'B-treatment', 'I-treatment']

        # # Replace values in the list with their corresponding IDs
        # mapped_ids = [mapping_dict[value] for value in values_list]

        # print("Mapped IDs:", mapped_ids)
        con_files_root = os.listdir("../Data/concept_assertion_relation_training_data/partners/concept")
        txt_files_root = os.listdir("../Data/concept_assertion_relation_training_data/partners/txt")
        final_data =[]
        for file in txt_files_root:
            print(file)
            con_file_name = file.split('.txt')[0]+'.con'
            if con_file_name in con_files_root:
                f = open("../Data/concept_assertion_relation_training_data/partners/txt/"+file,'r')
                lines = f.readlines()
                filename_con = "../Data/concept_assertion_relation_training_data/partners/concept/" + con_file_name
                conceptDict = self.computeConceptDict(filename_con)
                # print(conceptDict, "\n\n****************************************************")
                # break
                for line in range(0 ,len(lines)):
                    words =  str(lines[line]).split()
                    orginial_wordsList =  str(lines[line]).split()
                    IOBwordList= words
                    lineNumber= line+1 # Line number starts with 1
                    IOBwordList=self.prepareIOB_wordList(lineNumber,IOBwordList,conceptDict)
                    mapped_ids = [mapping_dict[value] for value in IOBwordList]
                    # print(type(orginial_wordsList))
                    final_data.append([orginial_wordsList, IOBwordList, mapped_ids])


        df = pd.DataFrame(final_data, columns=['tokens','tags','ner_tags'])
        # df.to_csv("../Data/concept_assertion_relation_training_data/partners/concept_data_4.csv",index=False)
        cwd  = os.getcwd()
        df.to_csv(os.path.join(cwd, f"{self.preprocessed_data_path}", "concept_data_final.csv", index=False))

    def load_concept_i2b2_data(self):
        cwd  = os.getcwd()
        path_beth_and_partners = os.path.join(cwd, f"{self.preprocessed_data_path}", "concept_data_final.csv")
        path_test = os.path.join(cwd, f"{self.preprocessed_data_path}", "concept_data_test.csv")
        beth_and_partners = pd.read_csv(path_beth_and_partners)
        test_data = pd.read_csv(path_test)
        return beth_and_partners, test_data