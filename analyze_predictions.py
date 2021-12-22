import os 
import pandas as pd
import string
import re
import json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def rid_of_specials(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        text = text.replace("<extra_id_2>", "")
        text = text.replace("<extra_id_3>", "")
        return text

    return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

log_dir = 'log'
prefix = 'IL_1e-4'
lst = os.listdir(log_dir)
lst.sort()
is_first = True
first = {}

for l in lst:
    if prefix in l:
        fname = log_dir + '/' + l
        df = pd.read_csv(fname, header=None)
        total_correct = 0
        same_correct = 0
        for index, row in df.iterrows():
            correct=False
            query = str(row[0])
            truth = str(row[1])
            pred = str(row[2])
            if normalize_answer(truth)==normalize_answer(pred):
                correct=True
                total_correct+=1
            if is_first and correct:
                first[index] = [query, pred]
                same_correct+=1
            else:
                if index in first:
                    first[index].append(pred)
                    if correct:
                        same_correct+=1
        print(total_correct , same_correct, total_correct-same_correct)
        is_first = False

entries = []
for key in first:
    identical = True
    row = first[key]
    rows = row[1:]
    for i in range(len(rows)):
        if rows[i]!=rows[0]:
            identical=False 
    if not identical:
        entries.append(row)

pd.DataFrame(entries).to_csv(prefix+'.csv', index=False)