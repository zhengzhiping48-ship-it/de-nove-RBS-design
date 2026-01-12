#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# ==== 手动设置输入与输出文件路径 ====
input_fasta_path = "DYTN-1 plasmid unnamed.fasta"   # 基因组 fasta 文件
list_xlsx_path = "list.xlsx"                     # Excel 文件，包含序列位置
output_txt_path = "output_p.txt"                   # 输出文本文件
show_detail = False                              # 是否在序列 ID 中显示位置信息

# ==== 读取基因组序列（假设只有一条染色体） ====
sequence = ""
with open(input_fasta_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('>'):
            continue
        sequence += line

# ==== 读取 Excel 文件 ====
df = pd.read_excel(list_xlsx_path)

# 确保列名正确
# Excel 文件必须包含：Name, Start, End, Strand
df.columns = ["name", "start", "end", "strand"]

# ==== 反向互补函数 ====
def rev(seq):
    base_trans = {'A':'T', 'C':'G', 'T':'A', 'G':'C',
                  'a':'t', 'c':'g', 't':'a', 'g':'c'}
    return ''.join(base_trans.get(b, b) for b in reversed(seq))  # 非ACGT保持不变

# ==== 提取序列并输出 ====
with open(output_txt_path, 'w', encoding='utf-8') as out:
    for _, row in df.iterrows():
        start = int(row["start"]) - 1  # 转为 0-based index
        end = int(row["end"])
        strand = row["strand"]
        name = row["name"]

        subseq = sequence[start:end]

        if strand == '+':
            seq_to_write = subseq
        else:
            seq_to_write = rev(subseq)

        if show_detail:
            out.write(f">{name} [{start+1}-{end} {strand}]\n")
        else:
            out.write(f">{name}\n")

        out.write(seq_to_write + "\n")

print("序列提取完成！输出文件为:", output_txt_path)
