import re
import ast
import matplotlib.pyplot as plt
import json


def count_render_results(file_path):
    total = 0
    success_true = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "render_result:" in line:
                total += 1
            if "A0100 has been shot down!" in line or 'success": True' in line:
                success_true += 1

    print(f"📦 render_result 总次数: {total}")
    print(f"✅ success=True 次数: {success_true}")
    if total > 0:
        print(f"📈 成功率: {success_true / total:.2%}")
        print(1-success_true / total)
    else:
        print("⚠️ 没有找到 render_result 行")

# 使用示例
log_file_path = "train/result/train_dodge.log"  # 替换成你的日志路径
count_render_results(log_file_path)



# def safe_parse_result(block):
#     try:
#         # 替换 Python 风格为 JSON 合法格式
#         block = block.replace("None", "null").replace("True", "true").replace("False", "false")
#
#         # 处理 numpy 的 array(...) 表达形式为 JSON 数组
#         block = re.sub(
#             r"array\(\s*\[(.*?)\]\s*\)",
#             lambda m: "[" + " ".join(m.group(1).replace("\n", " ").split()) + "]",
#             block,
#             flags=re.DOTALL
#         )
#
#         # 删除数组中数字后多余的空格（如 1. , → 1.0）
#         block = re.sub(r"(\d)\s*\.", r"\1.0", block)
#
#         # 将所有 key 和字符串值用双引号包裹（避免不合法引号）
#         block = re.sub(r"(?<=[:,{])\s*'([^']*)'\s*:", r'"\1":', block)  # keys
#         block = re.sub(r":\s*'([^']*)'", r':"\1"', block)              # values
#
#         return json.loads(block)
#     except Exception as e:
#         print(f"[!] fallback ast 失败: {e}\n原始内容:\n{block}\n")
#         try:
#             return ast.literal_eval(block)
#         except Exception as e2:
#             print(f"[!] ast 解析失败: {e2}\n")
#             return None
#
# def parse_render_result_line(line):
#     if "render_result:" in line:
#         try:
#             match = re.search(r"render_result:\s*(\{.*\})", line)
#             if match:
#                 result_dict = ast.literal_eval(match.group(1))
#                 return result_dict.get("success", None)
#         except Exception as e:
#             print("❌ 解析失败:", e)
#
#     return None
#
# def analyze_render_results(file_path, window_size=100):
#     total = 0
#     success_count = 0
#     sliding_window = []
#     success_rates = []
#
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             success = parse_render_result_line(line)
#             if success is not None:
#                 total += 1
#                 success_count += int(success)
#                 sliding_window.append(int(success))
#                 if len(sliding_window) == window_size:
#                     rate = sum(sliding_window) / window_size
#                     success_rates.append(rate)
#                     sliding_window = []  # 重置窗口
#
#     # 处理最后不足window_size的一段（可选）
#     if sliding_window:
#         rate = sum(sliding_window) / len(sliding_window)
#         success_rates.append(rate)
#
#     print(f"📊 总 render_result 条数: {total}")
#     print(f"✅ 成功次数 (success=True): {success_count}")
#     print(f"📈 成功率: {success_count / total:.2%}")
#
#     return success_rates
#
# def plot_success_rate(success_rates, window_size=100):
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, len(success_rates) + 1), success_rates, marker="o")
#     plt.xlabel(f"Window (每 {window_size} 条)")
#     plt.ylabel("Success Rate")
#     plt.title(f"Success Rate 每 {window_size} 条变化趋势")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# # 📝 设置你的日志文件路径
# log_file_path = "train/result/train_dodge3.log"  # 替换为实际文件路径
#
# # 🚀 运行分析
# rates = analyze_render_results(log_file_path, window_size=100)
#
# # 📊 画图
# plot_success_rate(rates, window_size=100)
