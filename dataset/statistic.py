import pandas as pd
HI = pd.read_csv('/home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/data/dataset/NHT/H5_NHT_HA.csv')
# HI = pd.read_csv('/home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/data/dataset/NHT/H3N2_full_HA.csv')
HI_unique_seqs = set(HI['seq'].tolist())
print(f'HI数据中序列的条数: {len(HI_unique_seqs)}')

def read_fasta_to_set(file_path):
    count = 0
    sequences = set()  # 初始化一个空集合
    current_seq = []   # 用于暂存当前正在读取的序列片段

    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除首尾的空格和换行符
            if not line:
                continue
            
            # 如果遇到 '>' 说明是新的一条记录的开始
            if line.startswith('>'):
                # 如果 current_seq 里面有内容，说明上一条序列读完了，把它加入 set
                if current_seq:
                    tmps = "".join(current_seq)
                    if tmps in HI_unique_seqs:
                        count += 1
                    sequences.add(tmps)
                    current_seq = []  # 清空暂存器，准备读取下一条
            else:
                # 不是以 '>' 开头，说明是序列内容，追加到暂存器中
                current_seq.append(line)
        
        # 循环结束后，别忘了把最后一条序列也加入 set
        if current_seq:
            tmps = "".join(current_seq)
            if tmps in HI_unique_seqs:
                        count += 1
            sequences.add(tmps)
        print(count)
    return sequences

# ====== 使用方法 ======
file_name = "/home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/data/dataset/all_seq/HA1_H5.fasta"  # 替换成你的 FASTA 文件路径，比如 "data/H5N6.fasta"
# file_name = "/home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/data/dataset/all_seq/H3N2_all.fasta"  # 替换成你的 FASTA 文件路径，比如 "data/H5N6.fasta"
unique_seqs = read_fasta_to_set(file_name)

print(f"成功读取！去重后共有 {len(unique_seqs)} 条不同的序列。")
# print(unique_seqs) # 如果需要查看具体内容可以取消注释

print(f'取交集后得到多少条：{len(HI_unique_seqs.intersection(unique_seqs))}')