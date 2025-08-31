services = [
    {'source': 9, 'destination': 7, 'rate': '100', 'possible_paths': [[9, 7], [9, 0, 7], [9, 6, 7], [9, 2, 6, 7], [9, 3, 6, 7], [9, 4, 0, 7], [9, 4, 6, 7], [9, 8, 6, 7], [9, 0, 4, 6, 7], [9, 2, 1, 0, 7]]},
    {'source': 5, 'destination': 9, 'rate': '10', 'possible_paths': [[5, 2, 9], [5, 3, 9], [5, 4, 9], [5, 2, 6, 9], [5, 3, 6, 9], [5, 3, 8, 9], [5, 4, 0, 9], [5, 4, 6, 9], [5, 4, 8, 9], [5, 2, 1, 0, 9]]}
]

# 提取所有路径并格式化，同时保留 source 和 destination 信息
all_paths = []
for service in services:
    for path in service['possible_paths']:
        all_paths.append({
            'odu_size': service['rate'],
            'path': path,
            'source': service['source'],
            'destination': service['destination']
        })

# 生成所有组合，排除自身组合和相同 source、destination 的组合
combinations = []
for i in range(len(all_paths)):
    for j in range(len(all_paths)):
        # 排除自身组合
        if i == j:
            continue
        # 排除相同 source 和 destination 的组合
        if all_paths[i]['source'] == all_paths[j]['source'] and all_paths[i]['destination'] == all_paths[j]['destination']:
            continue
        # 排除重复组合（确保 (A, B) 和 (B, A) 只保留一个）
        if j > i:
            combinations.append((all_paths[i], all_paths[j]))

# 删除组合中的 source 和 destination 字段
final_combinations = []
for combo in combinations:
    combo_cleaned = (
        {'odu_size': combo[0]['odu_size'], 'path': combo[0]['path']},
        {'odu_size': combo[1]['odu_size'], 'path': combo[1]['path']}
    )
    final_combinations.append(combo_cleaned)

# 输出结果
for combo in final_combinations:
    print(combo)