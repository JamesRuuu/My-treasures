1.read data，查阅格式/数据/特征列
2.回顾问题，取消的因素，y_label = is_canceled，去找特征件间的相似度
3.画图，柱图/折线，单一线形关系
4.特征间相似度 apriori/minhash（仅针对离散型数据）
先one-hot编码再求相似，jacarrd距离（把文字型分类特征做one-hot encode）
5.算相似度有很多方式，画图（直观），apriori（代码少），常用于推荐算法
