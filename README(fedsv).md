# Mitigating irrelevant clients in federated learning by Shapley value

fedavg, fedprox, scaffold 是三种用来当baseline的联邦学习算法。
fedsv_mc,fedsv_neyman,fedsv_neyman_original 只实现了对聚合时客户端参数权重的调整，它们的区别在于计算Shapley值的方法不同。

##要改进地方
*shapley计算
*选择客户端概率

 