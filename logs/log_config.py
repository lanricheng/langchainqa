import logging
from config import CFG

# 创建一个名为root的logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logging.basicConfig(format=LOG_FORMAT, datefmt='%a, %d %b %Y %H:%M:%S', filename=CFG.log_path, filemode='w')

# 创建一个文件处理器，将日志输出到文件
file_handler = logging.FileHandler(CFG.log_path)
file_handler.setLevel(logging.DEBUG)

# 创建一个控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 将格式器添加到处理器中
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到logger中
logger.addHandler(file_handler)
logger.addHandler(console_handler)
