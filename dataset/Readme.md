### HA文件中的数据列（H1N1_N2_NHT_HA.csv）

- `index`

  毒株索引号

- `name`

  毒株全称

- `location`

  毒株获取地点

- `id`

  毒株编码（不同年份的编码可能重复）

- `year`

  毒株获取年份

- `seq`

  毒株的HA序列

### HI文件中的数据列（H1N1_N2_NHT_HI.csv）

- `at_index`

  抗原毒株索引号

- `sr_index`

  血清毒株索引号

- `max_year`

  抗原-血清对中较大的年份

- `min_year`

  抗原-血清对中较小的年份

- `distance`

  抗原-血清对的抗原距离，用于回归任务

- `class`

  抗原-血清对的抗原性关系（~~抗原距离小于2认为抗原相似，标签为1；抗原距离大于等于2认为抗原漂移，标签为0）~~（抗原距离小于2认为抗原相似，标签为0；抗原距离大于等于2认为抗原漂移，标签为1）

  NHT距离的阈值是2

### 数据集

- `H1N1_N2.fasta`

  H1抗原进化数据集


- `H3N2.fasta`

  H3抗原进化数据集


- `H3N2_smith.fasta`

  smith数据集中的序列，序列数量少，年份跨度大

- `H5.fasta`

  H5抗原进化数据集

- `H5_NHT_HA.csv`、`H5_NHT_HI.csv`、`H5_AHT_HA.csv`、`H5_AHT_HI.csv`

  H5流感`抗原性/抗原变异`预测数据集，数据来自农科院。NHT距离不对称，AHT距离对称。2025年4月22前的数据集只提供了NHT。

* `H1N1_N2_NHT_HA.csv`、`H1N1_N2_NHT_HI.csv`、`H1N1_N2_AHT_HA.csv`、`H1N1_N2_AHT_HI.csv`

  H1流感`抗原性/抗原变异`预测数据集，病毒年份集中在2003-2024。NHT距离不对称，AHT距离对称。2025年4月22前的数据集只提供了NHT。

* `H3N2_NHT_HA.csv`、`H3N2_NHT_HI.csv`、`H3N2_AHT_HA.csv`、`H3N2_AHT_HI.csv`

  H3流感`抗原性/抗原变异`预测数据集，病毒年份集中在2003-2024。NHT距离不对称，AHT距离对称。2025年4月22前的数据集只提供了NHT。

* `H3N2_smith_NHT_HA.csv`、`H3N2_smith_NHT_HI.csv`、`H3N2_smith_AHT_HA.csv`、`H3N2_smith_AHT_HI.csv`

  H3流感`抗原性/抗原变异`预测数据集，病毒年份集中在1968-2003，这个数据集是从science（2004）的一篇文章中整理出来的，病毒的年份集中在1968-2003。NHT距离不对称，AHT距离对称。2025年4月22前的数据集只提供了NHT。

  **注：**

  1. ~~抗原进化数据集中的序列没有做预处理，序列中可能存在不确定氨基酸和较多的gap，可以根据下面的代码对序列进行预处理~~抗原进化数据集中的序列现在按照下面的代码进行了预处理

     ```
     def invalid_gap(seq: str) -> bool:
         # 询问孔老师这里gap的阈值设置为多少比较合适
         if 'X' in seq or 'B' in seq or 'Z' in seq or 'J' in seq :
             return True
         if seq[0] == "-" or seq[-1] == "-":
             return True
         # gap占比超过10%需要删掉
         if seq.count('-') * 10 > len(seq):
             # print("----------------")
             return True
         return False
     ```
  
  2. 所有的数据集中的序列数据没有做去重
  
  3. 对于毒株a和血清b，**NHT**距离计算公式为$D_{ab} = log_2(T_{bb}) - log_2(T_{ab})$，其中$T_{bb}$是毒株b和血清b的滴度值，$T_{ab}$是毒株a和血清b的滴度值。对于毒株a和毒株b，**AHT**的距离公式为$D_{ab} = \sqrt{\frac{T_{aa}\times T_{bb}}{T_{ab}\times T_{ba}}}$，其中$T_{aa}$是毒株a和血清a的滴度值，$T_{bb}$是毒株b和血清b的滴度值，$T_{ab}$是毒株a和血清b的滴度值，$T_{ba}$是毒株b和血清a的滴度值。
  
  
  
  
  
  
  
  