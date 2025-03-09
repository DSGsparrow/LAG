# -*- coding: utf-8 -*-
import logging


class Test:
    def train(self):
        logging.info('info train')

    def eval(self):
        logging.info('info eval')

        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.DEBUG,
                            filename='test.log',
                            filemode='w')

import logging
import os

# 假设 run_dir 是日志目录
run_dir = "./logs"  # 替换为你的 run_dir
os.makedirs(run_dir, exist_ok=True)  # 确保目录存在

log_file = os.path.join(run_dir, "run.log")  # 日志文件路径

# 创建 Logger
logger = logging.getLogger()  # 获取全局 logger
logger.setLevel(logging.INFO)  # 设定最低记录级别

# 清除已存在的 handlers，防止重复添加
logger.handlers.clear()

# 创建终端 Handler（StreamHandler）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设定终端输出级别

# 创建文件 Handler（FileHandler）
file_handler = logging.FileHandler(log_file, mode="a")  # "a" 追加模式
file_handler.setLevel(logging.INFO)  # 设定文件记录级别

# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加 handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 测试日志
logging.info("这条信息会同时输出到终端和 run.log 文件！")


t = Test()
t.train()
t.eval()


# logging.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logging.info('info级别，一般用来打印一些正常的操作信息')
# logging.warning('waring级别，一般用来打印警告信息')
# logging.error('error级别，一般用来打印一些错误信息')
# logging.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')
