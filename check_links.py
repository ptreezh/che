import re

# 读取HTML文件
with open('D:/AIDevelop/che_project/cognitive_heterogeneity_experiment.html', 'r', encoding='utf-8') as file:
    content = file.read()

# 查找所有链接
links = re.findall(r'href=[\'\"]([^\'\"]+)', content)
agentpsy_links = [link for link in links if 'agentpsy' in link]

print("所有链接:")
for i, link in enumerate(links, 1):
    print(f"{i}. {link}")

print(f"\nAgentPsy相关链接:")
for i, link in enumerate(agentpsy_links, 1):
    print(f"{i}. {link}")

print(f"\n总共找到 {len(links)} 个链接")
print(f"其中 {len(agentpsy_links)} 个指向agentpsy.com")