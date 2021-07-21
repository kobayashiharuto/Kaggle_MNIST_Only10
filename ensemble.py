import pandas as pd
import statistics

result1 = pd.read_csv('result/result_9946.csv')
result1 = result1['Label'].to_list()

result2 = pd.read_csv('result/result_9944.csv')
result2 = result2['Label'].to_list()

result3 = pd.read_csv('result/result_2.csv')
result3 = result3['Label'].to_list()

acc = 0
results = []
for index, (a, b, c) in enumerate(zip(result1, result2, result3)):
    x = statistics.mode([a, b, c])
    results.append([index + 1, x])

df = pd.DataFrame(results, columns=['ImageId', 'Label'])
df.to_csv('ensemble_result.csv', index=False)
