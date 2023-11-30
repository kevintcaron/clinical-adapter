"""
you will have to install the ollowing to run the file
pandas
spacy
and download spacy weights  using this command after you install spacy - python -m spacy download en_core_web_lg, (sm, md, lg)
"""

import pandas as pd
import os
import spacy
con_files_root = os.listdir("clinical-adapter/Data/concept_assertion_relation_training_data/beth/concept")
txt_files_root = os.listdir("clinical-adapter/Data/concept_assertion_relation_training_data/beth/txt")
print(len(con_files_root), len(txt_files_root))
# 74 75
for file in txt_files_root:
    if file.split('.txt')[0]+'.con' not in con_files_root:
        print(file)
# .DS_Store
# desktop.ini
nlp = spacy.load('en_core_web_lg')
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
        # print "start",start_idx
        # print "end",end_idx

        # print "line Number, start_idx,end_idx, iobclass",iobLineNo,start_idx,end_idx,iob_class
        # line_dict.update({iobLineNo:start_idx+"-"+end_idx+"-"+iob_class})

        if iobLineNo in line_dict.keys():
                # append the new number to the existing array at this slot
                # print "Found duplicate line number....."
                line_dict[iobLineNo].append(start_idx+"-"+end_idx+"-"+iob_class)
        else:
                # create a new array in this slot
                line_dict.update({iobLineNo:[start_idx+"-"+end_idx+"-"+iob_class]})
    #
    # for k,v in line_dict.iteritems():
    #     print k,v

    return line_dict

def prepareIOB_wordList(wordList,lineNumber,IOBwordList,conceptDict):
    # print "Line Number",lineNumber
    # print "Word- List ",wordList

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
         # print "IOB-  List ",IOBwordList
         # print "These Lines have ZERO IOB tags",IOBwordList
         # print "IOB Tag list ",iobTagList
    return IOBwordList




# Outfile = open("clinical-adapter/Data/concept_assertion_relation_training_data/beth/concept_data.txt", "a")
linecounter=0
final_data =[]
for file in txt_files_root:
    print(file)
    con_file_name = file.split('.txt')[0]+'.con'
    if con_file_name in con_files_root:
        f = open("clinical-adapter/Data/concept_assertion_relation_training_data/beth/txt/"+file,'r')
        lines = f.readlines()
        filename_con = "clinical-adapter/Data/concept_assertion_relation_training_data/beth/concept/" + con_file_name
        conceptDict = computeConceptDict(filename_con)
        print(conceptDict, "\n\n****************************************************")
        # break
        for line in range(0 ,len(lines)):
            words =  str(lines[line]).split()
            orginial_wordsList =  str(lines[line]).split()
            POS_word_TAGS = nlp(lines[line])
            linecounter+=1
            # print(words)
            # break
            IOBwordList= words
            # print words
            lineNumber= line+1 # Line number starts with 1
            #Prepare the IOB word list
            IOBwordList=prepareIOB_wordList(words,lineNumber,IOBwordList,conceptDict)

            for w in range(0,len(words)):
                # conllfileContent= orginial_wordsList[w] + "\t" + IOBwordList[w] +"\n"
                data = [file.split('.txt')[0], line, orginial_wordsList[w], IOBwordList[w], POS_word_TAGS[w].pos_, POS_word_TAGS[w].tag_, POS_word_TAGS[w].shape_, POS_word_TAGS[w].is_alpha, POS_word_TAGS[w].is_stop]
                # token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop
                # print conllfileContent
                # Outfile.write(conllfileContent)
                # Outfile.write("\n")
                final_data.append(data)


                # [[ytd,wkeg,gff],[sfdh,akjg,ashd]]


# print(final_data)
df = pd.DataFrame(final_data, columns=['record_id','line_no','word','NER_TAG','POS','TAG','shape','is_alpha','is_stop'])
df.to_csv("clinical-adapter/Data/concept_assertion_relation_training_data/beth/concept_data.csv",index=False)
