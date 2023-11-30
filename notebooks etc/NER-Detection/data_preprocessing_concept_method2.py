"""
you will have to install the following to run the file
pandas
"""

import pandas as pd
import os
con_files_root = os.listdir("../Data/concept_assertion_relation_training_data/partners/concept")
txt_files_root = os.listdir("../Data/concept_assertion_relation_training_data/partners/txt")
print(len(con_files_root), len(txt_files_root))
# 74 75
for file in txt_files_root:
    if file.split('.txt')[0]+'.con' not in con_files_root:
        print(file)
# .DS_Store
# desktop.ini
def computeConceptDict(conFilePath):
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

def prepareIOB_wordList(lineNumber,IOBwordList,conceptDict):

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

final_data =[]
for file in txt_files_root:
    print(file)
    con_file_name = file.split('.txt')[0]+'.con'
    if con_file_name in con_files_root:
        f = open("../Data/concept_assertion_relation_training_data/partners/txt/"+file,'r')
        lines = f.readlines()
        filename_con = "../Data/concept_assertion_relation_training_data/partners/concept/" + con_file_name
        conceptDict = computeConceptDict(filename_con)
        # print(conceptDict, "\n\n****************************************************")
        # break
        for line in range(0 ,len(lines)):
            words =  str(lines[line]).split()
            orginial_wordsList =  str(lines[line]).split()
            IOBwordList= words
            lineNumber= line+1 # Line number starts with 1
            IOBwordList=prepareIOB_wordList(lineNumber,IOBwordList,conceptDict)
            mapped_ids = [mapping_dict[value] for value in IOBwordList]
            # print(type(orginial_wordsList))
            final_data.append([orginial_wordsList, IOBwordList, mapped_ids])


df = pd.DataFrame(final_data, columns=['tokens','tags','ner_tags'])
df.to_csv("../Data/concept_assertion_relation_training_data/partners/concept_data_4.csv",index=False)


import pandas as pd

# Assuming you have loaded the CSV file into a DataFrame named df
# df = pd.read_csv('your_file.csv')
import ast
# Iterate over the rows
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

df_to_remove = df[df['ner_tags'].apply(lambda tags: all(tag == 0 for tag in ast.literal_eval(tags)))].sample(frac=0.8)
df = df.drop(df_to_remove.index)

# count0=0
# count=0
# for index, row in df.iterrows():
#     for i in ast.literal_eval(row['tags']):
#         if i=='O':
#             count0+=1
#         else:
#             count+=1