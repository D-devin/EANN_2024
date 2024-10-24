import csv
import requests
from bs4 import BeautifulSoup

# 读取CSV文件
input_csv = r'E:\fakenews\EANN_2024\Data\weixin\train\train.csv'  # 替换为你的CSV文件路径
output_csv = r'new_train.csv'  # 替换为你希望保存的输出CSV文件路径

# 读取CSV文件并提取URL和行索引
urls = []
with open(input_csv, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for index, row in enumerate(reader):
        urls.append((index, row['News Url']))

    # 定义一个函数来提取网页文本内容
def get_webpage_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败，抛出HTTPError异常
        soup = BeautifulSoup(response.content, 'html.parser')
        # 提取网页中的文本内容，这里可以根据实际情况调整提取方式
        # 例如，可以提取<p>标签内的文本，或者整个页面的纯文本
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

    # 创建一个列表来保存修改后的行数据


modified_rows = []

# 遍历URL列表，获取网页文本并更新行数据
for index, url in urls:
    row = {key: '' for key in
           csv.DictReader(open(input_csv, mode='r', encoding='utf-8')).fieldnames}  # 创建一个空字典，包含所有列名
    with open(input_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, r in enumerate(reader):
            if i == index:
                # 获取当前行的数据
                for key, value in r.items():
                    row[key] = value
                    # 获取网页文本内容
                webpage_text = get_webpage_text(url)
                if webpage_text:
                    # 如果需要替换原有的URL列，可以将"News Url"替换为网页文本
                    row['News Url'] = webpage_text  # 如果要替换URL列
                    #row['Text Content'] = webpage_text  # 如果要添加新列
                break
    modified_rows.append(row)

# 将修改后的行数据写入新的CSV文件
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = csv.DictReader(open(input_csv, mode='r', encoding='utf-8')).fieldnames  # 包含所有原列名和新添加的列名
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(modified_rows)

print("CSV文件已成功更新并保存到", output_csv)
