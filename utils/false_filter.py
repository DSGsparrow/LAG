import json


def filter_failed_experiments(input_file, output_file):
    """
    读取 JSON 文件，筛选出 `success=false` 的实验数据，并保存到新的 JSON 文件中。

    :param input_file: 原始 JSON 文件路径
    :param output_file: 结果 JSON 文件路径
    """
    failed_experiments = []

    # 逐行读取 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)  # 解析 JSON 行
            if not entry.get("success", True):  # 仅保留 success=False 的数据
                failed_experiments.append(entry)

    # 将筛选后的数据写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in failed_experiments:
            json.dump(item, f)
            f.write("\n")  # 每个 JSON 对象单独占一行

    print(f"筛选完成，{len(failed_experiments)} 条失败实验数据已保存到 {output_file}")


# **使用示例**
input_json = "../test_result/dodge_test/evaluated_results_all.json"  # 你的原始实验 JSON 文件
output_json = "../test_result/dodge_test/fail_results.json"  # 筛选后的 JSON 文件

filter_failed_experiments(input_json, output_json)
