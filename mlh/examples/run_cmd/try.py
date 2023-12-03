import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Example of parsing boolean values.")

# 添加布尔类型的参数
# 如果命令行中出现了'--feature', 那么 args.feature 将被设置为 True
parser.add_argument('--feature', action='store_true', help='Enable feature')

# 如果命令行中出现了'--no-feature', 那么 args.feature 将被设置为 False
parser.add_argument('--no-feature', action='store_false', help='Disable feature')

# 默认值通常设置为最常用的值或系统默认值
parser.set_defaults(feature=True)

# 解析命令行参数
args = parser.parse_args()

# 输出解析后的参数值
print(f"Feature is enabled: {args.feature}")
