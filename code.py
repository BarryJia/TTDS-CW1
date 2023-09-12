from __future__ import division
import json
import string
import xml.dom.minidom
import math
import re
import time

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Path prefix of the file to be saved
filePreAddress = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/results/'
# Path of the collection file (.xml format)
fileRead = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/CW1collection/trec.5000.xml'
# Path of the stop words file
fileStopwords = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/englishST.txt'
# Path of the query file (boolean, phrase, proximity)
booleanQueries = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/CW1collection/queries.boolean.txt'
# Path of the inverted index file
fileToLoad = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/results/index.txt'
# Path of the Ranked IR query file
rankedQueries = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/CW1collection/queries.ranked.txt'
# Path of the result file (for Ranked IR only)
TFIDFresults = '/Users/jiashichao/Desktop/Edinburgh/Text_Technologies_for_Data_Science/CW1/results/results.ranked.txt'

# open the .xml file and read it by ID and TEXT.
# open the local stopwords list and load it as a list.
# for each TEXT (document) in the collection, preprocess it and store it into a dict.
# write it into a .txt file, and return the ID list.
def IndexCreation():
    idlist = []
    dict = {}
    documentList = []
    stopwordsli = []
    dom = xml.dom.minidom.parse(fileRead)
    root = dom.documentElement
    headline = root.getElementsByTagName('HEADLINE')
    text = root.getElementsByTagName('TEXT')
    for i in range(len(text)):
        documentList.append(headline[i].firstChild.data + ' ' + text[i].firstChild.data)
    idTotal = root.getElementsByTagName('DOCNO')
    for i in idTotal:
        idlist.append(i.firstChild.data)
    regEx = re.compile('\W')
    with open(fileToLoad, "w") as fr, open(fileStopwords) as fstop:
        temp_lis = {}
        for item in range(len(documentList)):
            temp_lis.update({idlist[item] : documentList[item]})
        for lines in fstop:
            stopwordsli.append(lines.rstrip())
        for element in temp_lis:
            originText = temp_lis[element]
            lineword = regEx.split(originText)
            newLineword = []
            for lw in lineword:
                if lw not in stopwordsli:
                    newLineword.append(lw)
            finallineword = [ps.stem(word) for word in newLineword]
            doc_index = element
            for word in finallineword:
                if len(word)>0:
                    if word not in dict:
                        dict[word] = {doc_index: [i + 1 for i, x in enumerate(finallineword) if x == word]}
                    else:
                        dict[word].update({doc_index: [i + 1 for i, x in enumerate(finallineword) if x == word]})
        for x in sorted(dict):
            fr.write (str(x) + ":" + str(len(dict[x])))
            fr.write ("\n")
            for y in dict[x]:
                fr.write("\t")
                fr.write(str(y) + ":" + " ")
                fr.write(",".join([str(i) for i in dict[x][y]]))
                fr.write("\n")
            fr.write("\n")
    fr.close()
    print(len(dict))
    return idlist

# open the index file and read it by lines.
# add the appropriate symbols so that the string can be converted to dict. (also delete the term frequency.)
# load it as a dict and return it.
def LoadIndex(fileToLoad):
    preprocessResult = open(fileToLoad, 'r')
    preprocessDict = {}
    for line in preprocessResult:
        l = line.rstrip()
        if l:
            if l[0] != '\t':
                currentTerm = line.rstrip().rstrip(string.digits).replace(':', '')
            else:
                frequencyList = re.split(':', l.strip('\t'))
                keyValue = "{\"" + frequencyList[0] + "\"" + ":[" + frequencyList[1].strip() + "]}"
                currentDict = json.loads(keyValue)
                if currentTerm not in preprocessDict:
                    preprocessDict[currentTerm] = currentDict
                else:
                    preprocessDict[currentTerm].update(currentDict)
    preprocessResult.close()
    return preprocessDict

# first: extract the queries and remove the character in the beginning, append the query to a query list.
# second: for every query, decide the query is a Boolean query, Phrase query or Proximity query.
# store the retrieved result in a dict.
def SearchFunction(idlist, dict):
    resultDict = {}
    with open(fileToLoad) as fr,  open(booleanQueries) as fq, open(filePreAddress+"results.boolean.txt", "w") as frl:
        qlis = []
        j = 1
        for line in fq:
            queries = line.split(" ", 1)
            qlis.append(queries[1].rstrip())
        for query in qlis:
            retrievedocs = [[]]
            retrivindex = 0
            if "AND" in query or "OR" in query or "NOT" in query:
                words = re.split(" +(AND|OR) +", query)
                i = 0
                andor = 0
                while (i < len(words)):
                    if (words[i][0:3] == "NOT"):
                        nextword = words[i].split()[1]
                        nextword = nextword.lower()
                        nextword = ps.stem(nextword)
                        for x in dict:
                            if x == nextword:
                                docslist = list(set(idlist) - set(dict[x].keys()))
                                retrievedocs.insert(retrivindex, docslist)
                        retrivindex = retrivindex + 1
                        i = i + 1
                        continue
                    elif (words[i][0] == "\""):
                        phrasewords = words[i].replace('"', '')
                        phraselist = phrasewords.split()
                        phraselist = [phrase.lower() for phrase in phraselist]
                        phraselist = [ps.stem(phrase) for phrase in phraselist]
                        for x in dict:
                            if x == phraselist[0]:
                                for y in dict:
                                    if y == phraselist[1]:
                                        for docid in dict[x]:
                                            if docid in dict[y].keys():
                                                indexlist = [a+1 for a in dict[x][docid]]
                                                findlist = [b for b in dict[y][docid]]
                                                if len([c for c in indexlist if c in findlist]) != 0:
                                                    if (retrievedocs[retrivindex]):
                                                        retrievedocs[retrivindex].extend([docid])
                                                    else:
                                                        retrievedocs.insert(retrivindex, [docid])
                        retrivindex = retrivindex + 1
                        i = i + 1
                        continue
                    elif(words[i] == "AND"):
                        andor = 1
                        i = i + 1
                        continue
                    elif(words[i] == "OR"):
                        andor = 2
                        i = i + 1
                        continue
                    else:
                        word = words[i]
                        word = word.lower()
                        word = ps.stem(word)
                        for x in dict:
                            if x == word:
                                retrievedocs.insert(retrivindex, dict[x].keys())
                        retrivindex = retrivindex + 1
                        i = i + 1
                        continue
                finalist = []
                if (andor == 0):
                    finalist = retrievedocs[0]
                elif (andor == 1):
                    set1 = set(retrievedocs[0])
                    set2 = set(retrievedocs[1])
                    finalist = list(set1 & set2)
                elif (andor == 2):
                    finalist = list(set().union(retrievedocs[0], retrievedocs[1]))
                finalist = sorted(finalist)
                for fid in finalist:
                    if j not in resultDict:
                        resultDict[j] = [fid]
                    else:
                        resultDict[j].append(fid)
            elif (query[0] == "#"):
                regEx = re.compile('\W')
                beforelist = regEx.split(query)
                afterlist = [a for a in beforelist if a != '']
                proxindex = afterlist[0]
                proxlist = afterlist[1:]
                proxlist = [prh.lower() for prh in proxlist]
                proxlist = [ps.stem(prh) for prh in proxlist]
                for x in dict:
                    if x == proxlist[0]:
                        for y in dict:
                            if y == proxlist[1]:
                                for srdoc in dict[x]:
                                    if srdoc in dict[y].keys():
                                        indexlis = [a for a in dict[x][srdoc]]
                                        findlis = [b for b in dict[y][srdoc]]
                                        for a in indexlis:
                                            for b in findlis:
                                                if abs(int(a)-int(b)) <= int(proxindex):
                                                    if (retrievedocs[0]):
                                                        retrievedocs[0].extend([srdoc])
                                                    else:
                                                        retrievedocs.insert(0, [srdoc])
                finalist = sorted(list(set(retrievedocs[0])), key = lambda number:int(number))
                for fid in finalist:
                    if j not in resultDict:
                        resultDict[j] = [fid]
                    else:
                        resultDict[j].append(fid)
            elif (query[0] == "\""):
                phrasewords = query.replace('"', '')
                phraselist = phrasewords.split()
                phraselist = [prh.lower() for prh in phraselist]
                phraselist = [ps.stem(prh) for prh in phraselist]
                for x in dict:
                    if x == phraselist[0]:
                        for y in dict:
                            if y == phraselist[1]:
                                for srdoc in dict[x]:
                                    if srdoc in dict[y].keys():
                                        indexlis = [a+1 for a in dict[x][srdoc]]
                                        findlis = [b for b in dict[y][srdoc]]
                                        if len([c for c in indexlis if c in findlis]) != 0:
                                            if (retrievedocs[0]):
                                                retrievedocs[0].extend([srdoc])
                                            else:
                                                retrievedocs.insert(0, [srdoc])
                finalist = sorted(list(set(retrievedocs[0])), key = lambda number:int(number))
                for fid in finalist:
                    if j not in resultDict:
                        resultDict[j] = [fid]
                    else:
                        resultDict[j].append(fid)
            else:
                wor = query
                wor = wor.lower()
                wor = ps.stem(wor)
                for x in dict:
                    if x == wor:
                        retrievedocs[0] = dict[x].keys()
                finalist = sorted(retrievedocs[0], key = lambda number:int(number))
                for fid in finalist:
                    if j not in resultDict:
                        resultDict[j] = [fid]
                    else:
                        resultDict[j].append(fid)
            j = j + 1
        for x in resultDict:
            for y in resultDict[x]:
                frl.write(str(x)+','+str(y))
                frl.write('\n')
        frl.close()
        fr.close()
        fq.close()
        return resultDict

# for the specific document and term, return the TF result.
def TF(documentID, term, indexDict):
    for i in indexDict:
        if i == term:
            for j in indexDict[i]:
                if j == documentID:
                    return math.log10(len(indexDict[i][j]))

# for the specific term, return the IDF result.
def IDF(indexDict, term):
    for i in indexDict:
        if i == term:
            df = len(indexDict[i])
            result = math.log10(len(idlist)/df)
            return result

# according to the query, call TF() and IDF() functions to calculate the TFIDF result as a dict and return.
def TFIDFscore(query, indexDict):
    queryList = query.split()
    finalScore = {}
    for i in idlist:
        finalScore[i] = 0
    for q in queryList:
        for index in indexDict:
            if ps.stem(q.lower()) == index:
                idf = IDF(indexDict, index)
                for nextindex in indexDict[index]:
                    tf = TF(nextindex, index, indexDict)
                    finalScore[nextindex] = finalScore[nextindex] + ((1+tf)*idf)
    newScore = {}
    for eachScore in finalScore:
        if finalScore[eachScore] > 0:
            newScore[eachScore] = format(finalScore[eachScore], '.4f')
    return newScore

# preprocess the query, store it as a list and return.
def LoadQuery(queryToLoad, stopwords):
    queryText = open(queryToLoad)
    regEx = re.compile('\W')
    queryDict = {}
    for line in queryText:
        lineList = regEx.split(line)
        query = ""
        for term in lineList[1:]:
            if term.lower() not in stopwords:
                query = query + ps.stem(term.lower()) + " "
        queryDict[lineList[0]] = query
    print(queryDict)
    return queryDict

# call LoadQuery() function to get the query list after preprocessed.
# for each query in the list, call TFIDFscore() function to calculate the result.
# decide the length of the sorted result, write it into the .txt file.
def RankedIR(indexDict, stopwordslist):
    with open(TFIDFresults, 'w') as fresult:
        queryDict = LoadQuery(rankedQueries, stopwordslist)
        for query in queryDict:
            tfidf = TFIDFscore(queryDict[query], indexDict)
            if len(tfidf) >= 150:
                result = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:150]
            else:
                result = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
            for r in result:
                fresult.write(query+","+r[0]+","+str(r[1]))
                fresult.write('\n')
    fresult.close()

if __name__ == '__main__':
    start = time.time()
    idlist = IndexCreation()
    # print(idlist)
    indexDict = LoadIndex(fileToLoad)
    # SearchFunction(idlist, indexDict)
    stopwordslist = []
    with open(fileStopwords) as fstop:
        for lines in fstop:
            stopwordslist.append(lines.rstrip())
    RankedIR(indexDict, stopwordslist)
    # end = time.time()
    # print(end-start)