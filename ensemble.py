import pandas as pd
import statistics

result1 = pd.read_csv('result/result_9946.csv')
result1 = result1['Label'].to_list()

result2 = pd.read_csv('result/result_9944.csv')
result2 = result2['Label'].to_list()

result3 = pd.read_csv('result/result_9949_v7.csv')
result3 = result3['Label'].to_list()

result4 = pd.read_csv('result/result_9939_v8.csv')
result4 = result4['Label'].to_list()

result5 = pd.read_csv('result/result_v9.csv')
result5 = result5['Label'].to_list()

acc = 0
results = []
for index, (a, b, c, d, e) in enumerate(zip(result1, result2, result3, result4, result5)):
    x = statistics.mode([a, b, c, d, e])
    results.append([index + 1, x])

df = pd.DataFrame(results, columns=['ImageId', 'Label'])
df.to_csv('result/ensemble_result_4.csv', index=False)
