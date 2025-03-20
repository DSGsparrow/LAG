import json

def merge_multiple_json_files(file_list, output_file):
    """
    读取多个 JSON 文件，将其合并到一个新的 JSON 文件
    :param file_list: JSON 文件名列表
    :param output_file: 输出合并后的 JSON 文件路径
    """
    merged_data = []

    # 遍历文件列表，按顺序读取
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                merged_data.append(json.loads(line))  # 逐行加载 JSON 对象

    # 写入合并后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            json.dump(item, f)
            f.write("\n")  # 每行存储一个 JSON 对象

    print(f"合并完成，已保存至 {output_file}")

# **使用示例**
file1 = "./test_result/dodge_test/parsed_results1.json"
file2 = "./test_result/dodge_test/parsed_results2.json"
file3 = "./test_result/dodge_test/parsed_results3_total.json"

file_list = [file1, file2, file3]  # 你的 JSON 文件列表
output_json = "./test_result/dodge_test/parsed_results_all.json"  # 目标合并文件
merge_multiple_json_files(file_list, output_json)
