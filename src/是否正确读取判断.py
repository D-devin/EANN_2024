import pandas as pd
# 读取CSV文件
df = pd.read_csv('news_train.csv')
# 定义一个函数来检查文本内容是否包含无法读取的关键词
def check_content_for_errors(text):
    # 定义无法读取的关键词列表
    error_keywords = [
        "账号 屏蔽", "内容 无法 查看", "内容发布者 删除", "微信 公众 平台 运营 中心",
        "账号 迁移", "公众号 环境异常"
    ]
    # 检查文本中是否包含任何关键词
    for keyword in error_keywords:
        if keyword in text:
            return "1"
    return "2"
# 并创建新的'news_tag'列
df['news_tag'] = df['News Url'].apply(check_content_for_errors)

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('last_train.csv', index=False)

print('处理完成')