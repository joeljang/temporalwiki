import pandas as pd 
import random

dataset = pd.read_csv('data/wikipedia_0809_subset.csv')

total_line = len(dataset)
length = 10000
skip = sorted(random.sample(range(1,total_line+1),total_line-length))
dataset = pd.read_csv('data/wikipedia_0809_subset.csv', skiprows=skip)

dataset.to_csv('data/perplexity/wikipedia_0809_subset.csv')

# dataset = pd.read_csv('data/wikipedia_10_gpt2/part1.csv')
# dataset2 = pd.read_csv('data/wikipedia_10_gpt2/part2.csv')

# total_line2 = len(dataset)

# dataset3 = pd.concat([dataset,dataset2], ignore_index=True)

# total_line = len(dataset3)
# length = 10000
# skip = sorted(random.sample(range(1,total_line+1),total_line-length))
# skip1 = []
# skip2 = []
# for elem in skip:
#     if elem <= total_line2:
#         skip1.append(elem)
#     else: 
#         skip2.append(elem-total_line2)
# dataset = pd.read_csv('data/wikipedia_10_gpt2/part1.csv', skiprows=skip1)
# # dataset.to_csv('data/perplexity/wikipedia_10_gpt2')
# dataset2 = pd.read_csv('data/wikipedia_10_gpt2/part2.csv', skiprows=skip2)
# # dataset2.to_csv('data/perplexity/wikipedia_10_gpt2_2')

# dataset3 = pd.concat([dataset,dataset2], ignore_index = True, axis=0)
# dataset3 = dataset.append(dataset2)

# dataset3.to_csv('data/perplexity/wikipedia_10_gpt2.csv')