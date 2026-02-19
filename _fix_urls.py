"""Temporary script to fix API_BASE template literals in script.js"""
import re

path = r'Front/script.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace single-quoted '${API_BASE}/...' with backtick `${API_BASE}/...`
old_count = content.count("'${API_BASE}")
content = re.sub(r"'(\$\{API_BASE\}/[^']*)'", r'`\1`', content)
new_count = content.count("'${API_BASE}")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Fixed {old_count - new_count} single-quoted API_BASE references")
print(f"Remaining single-quoted: {new_count}")

# Verify backtick count
backtick_count = len(re.findall(r'`\$\{API_BASE\}', content))
print(f"Backtick template literals: {backtick_count}")
