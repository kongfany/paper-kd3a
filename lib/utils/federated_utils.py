from itertools import permutations, combinations
import torch


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def update_domain_weight(global_domain_weight, epoch_domain_weight, momentum=0.9):
    global_domain_weight = [round(global_domain_weight[i] * momentum + epoch_domain_weight[i] * (1 - momentum), 4)
                            for i in range(len(epoch_domain_weight))]
    return global_domain_weight


def federated_average(model_list,model_deputy_list, coefficient_matrix, batchnorm_mmd=True):#false
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    :param model_list:联邦平均所需的所有模型的列表。
    [0]:目标域的模型;[1:-1]源域模型
    :param coefficient_matrix:每个模型的系数以联邦平均、列表或一维np数组表示
    :param batchnorm_mmd: bool，如果为true，则使用batchnorm MMD
    :返回联邦平均后的模型列表

    """

    # (Pdb) p len(model_list)
    # 6
    # (Pdb) p coefficient_matrix
    # [0.0549, 0.0713, 0.2351, 0.0859, 0.2743, 0.2786]
    # (Pdb)
    if batchnorm_mmd:
        # 上传的是个性化模型
        dict_list = [it.state_dict() for it in model_list]# 每个模型的权重以字典形式存储
        dict_item_list = [dic.items() for dic in dict_list]# 每个权重的键值对，将其分别从每个模型的字典中取出

        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]# 每个元组，将其中所有模型的权重按照系数矩阵coefficient_matrix进行加权
            dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
            # (Pdb) p key_data_pair_list[0][0]
            # 'encoder.module.conv1.weight'
        # for model in model_list:
        #     model.load_state_dict(dict_list[0])
        # 接收的是副模型
        for model_deputy in model_deputy_list[1:]:
            model_deputy.load_state_dict(dict_list[0])
        model_list[0].load_state_dict(dict_list[0])
        # 将多个模型的权重按照一定的系数加权合并，并将合并后的权重重新加载到所有模型中
        # 首先，将所有模型的权重保存到一个列表dict_list中，每个模型的权重以字典形式存储。
        # 对于每个权重的键值对，将其分别从每个模型的字典中取出，保存到一个列表dict_item_list中。
        # 使用zip()函数将同一个键值对的不同模型的权重打包在一起，得到一个包含多个元组的列表key_data_pair_list。
        # 遍历key_data_pair_list，对于每个元组，将其中所有模型的权重按照系数矩阵coefficient_matrix进行加权，并求和，得到该键值对在合并后的权重字典中的值。
        # 将合并后的权重字典赋值给第一个模型的权重字典。
        # 最后，将第一个模型的合并后的权重加载到所有模型中，保证所有模型的权重都是相同的。

        # (Pdb) p model_list[1].state_dict().keys() == model_list[2].state_dict().keys()
        # True
    else:
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            parameter_list[0][1].data = sum(source_parameters)
            for parameter in parameter_list[1:]:
                parameter[1].data = parameter_list[0][1].data.clone()

# 在这段代码中，参数batchnorm_mmd用于控制使用不同的权重合并方法。当batchnorm_mmd为True时，会使用BatchNorm MMD（Maximum Mean Discrepancy）方法进行权重合并；当batchnorm_mmd为False时，会使用普通的权重加权平均方法进行合并。
#
# 在使用BatchNorm MMD方法时，先将每个模型的权重保存到一个列表中，并将每个权重的键值对分别提取出来。然后，针对每个键值对，将所有模型的权重按照系数矩阵进行加权求和，得到合并后的权重值。最后，将合并后的权重字典加载到所有模型中，以确保所有模型的权重一致。
#
# 而在不使用BatchNorm MMD方法时，首先将每个模型的参数使用named_parameters()函数提取出来，并将不同模型对应的参数打包在一起。然后，对于每组参数，将所有模型的参数按照系数矩阵进行加权求和，得到合并后的参数值，并将该值赋给第一个模型对应的参数。最后，将第一个模型的合并后的参数值加载到所有模型中，保证所有模型的参数一致。
#
# 因此，使用batchnorm_mmd=True时，会对模型的权重进行合并；而使用batchnorm_mmd=False时，会对模型的参数进行合并。这是两种不同的权重合并策略，根据具体情况选择适合的方法。

def knowledge_vote(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    : param火炬。张量knowledge_list:记录来自每个源域模型的知识
    :参数float confidence . gate:判断使用哪个样本的置信门
    :返回:consensus_confidence, consensus_knowledge consensus_knowledge_weight
    """
    # (Pdb) p knowledge_list.shape
    # torch.Size([128, 4, 10])
    max_p, max_p_class = knowledge_list.max(2)# 找到每个类别的最大值和对应的类别
    # max_p 128个数据对应的每个客户端的具体类别的最大值，每个源域中最大概率值的集合
    # max_p_class 最大值对应的位置，即类别是每个源域中对应最大概率值的类别索引。

    # tensor([[3.2791e-14, 1.0000e+00, 8.3796e-10, 6.0148e-12, 1.8025e-15, 4.5243e-13,
    #          7.0788e-19, 4.7752e-10, 2.5085e-10, 2.0582e-17],
    #         [1.4746e-03, 9.9649e-01, 1.9400e-03, 3.7293e-09, 4.1397e-05, 4.0905e-09,
    #          2.8378e-07, 4.5949e-05, 7.4853e-07, 8.3291e-06],
    #         [2.9592e-08, 9.9332e-01, 1.4715e-05, 7.5122e-11, 6.6619e-03, 3.1080e-11,
    #          1.4477e-07, 1.8239e-06, 7.8634e-10, 2.7150e-11],
    #         [2.0656e-04, 1.1220e-01, 2.2573e-02, 9.5399e-06, 4.1913e-04, 2.8473e-06,
    #          1.4350e-06, 8.6447e-01, 4.5728e-05, 7.4664e-05]], device='cuda:0')
    #
    # (Pdb) p knowledge_list[0].shape
    # torch.Size([4, 10])
    # p knowledge_list[0][0]
    # tensor([3.2791e-14, 1.0000e+00, 8.3796e-10, 6.0148e-12, 1.8025e-15, 4.5243e-13,
    #         7.0788e-19, 4.7752e-10, 2.5085e-10, 2.0582e-17], device='cuda:0')

    # (Pdb) p max_p.shape
    # torch.Size([128, 4])
    # (Pdb) p max_p_class.shape
    # torch.Size([128, 4])
    # (Pdb) p max_p[0]
    # tensor([1.0000, 0.9992, 0.8446, 0.8663], device='cuda:0')
    # (Pdb) p max_p_class[0]
    # tensor([1, 1, 4, 1], device='cuda:0')

    max_conf, _ = max_p.max(1)#
    # (Pdb) p max_conf.shape
    # torch.Size([128])
    # (Pdb) p max_conf[0]
    # tensor(1., device='cuda:0')
    max_p_mask = (max_p > confidence_gate).float().cuda()# 置信度判断
    # (Pdb) p max_p_mask.shape
    # torch.Size([128, 4])
    # (Pdb) p max_p_mask[0]
    # tensor([1., 1., 0., 1.], device='cuda:0')
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).cuda()
    # (Pdb) p consensus_knowledge.shape
    # torch.Size([128, 10])
    # 创建一个全零的张量 consensus_knowledge，
    # 形状为 (batch_size, num_classes)，用于累积共识知识
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # (Pdb) p p
        # tensor([1.0000, 0.9965, 0.9933, 0.8645], device='cuda:0')
        # (Pdb) p p_class
        # tensor([1, 1, 1, 7], device='cuda:0')
        # (Pdb) p p_mask
        # tensor([1., 1., 1., 0.], device='cuda:0')
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
            # 将 p 乘以 p_mask，如果 p_mask 中的元素为1，
            # 即 max_p 大于 confidence_gate，则保留原始值，否则置为0。
            # (Pdb) p p
            # tensor([1.0000, 0.9965, 0.9933, 0.0000], device='cuda:0')
        for source_idx, source_class in enumerate(p_class):
            # (Pdb) p p_class
            # tensor([1, 1, 1, 2], device='cuda:0')
            # (Pdb) p source_idx
            # 0
            # (Pdb) p source_class
            # tensor(1, device='cuda:0')
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
            # (Pdb) p consensus_knowledge.shape
            # torch.Size([128, 10])
            # 遍历每个源域中的类别索引 source_class，
            # 将 p 中的值累加到 consensus_knowledge 中对应的位置。
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    # # (Pdb) p consensus_knowledge.shape
    #             # torch.Size([128, 10])
    # (Pdb) p consensus_knowledge[0]
    # tensor([0.0000, 1.9853, 0.9043, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #         0.0000], device='cuda:0')
    # consensus_knowledge_conf 是每个样本中最大共识知识的值，
    # (Pdb) p consensus_knowledge_conf.shape
    # torch.Size([128])
    # (Pdb) p consensus_knowledge_conf[0]
    # tensor(1., device='cuda:0')
    # consensus_knowledge 是每个样本中最大共识知识的类别索引。
    # (Pdb) p consensus_knowledge.shape
    # torch.Size([128])
    # (Pdb) p consensus_knowledge[0]
    # tensor(1, device='cuda:0')
    consensus_knowledge_mask = (max_conf > confidence_gate).float().cuda()
    # 创建一个掩码 consensus_knowledge_mask，其中元素为1的位置表示对应位置的 max_conf 大于 confidence_gate，否则为0。
    # (Pdb) p consensus_knowledge_mask.shape
    # torch.Size([128])
    for idx, (conf_class, ck, mask) in enumerate(zip(max_conf_class, consensus_knowledge, consensus_knowledge_mask)):
        if conf_class != ck:
            mask[idx]=0
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    # 创建一个全零的张量 consensus_knowledge，形状为 (batch_size, num_classes)，
    # 并根据 consensus_knowledge 的类别索引将对应位置置为1，用于表示共识知识。

    # 使用张量的 scatter_() 方法，按照指定的维度和索引，将给定的值填充到张量中。
    # 在这里，我们将共识知识 consensus_knowledge 的类别索引 consensus_knowledge.view(-1, 1) 视为索引张量，
    # 将值为1的向量 [1] 填充到 consensus_knowledge 中对应位置的维度1上。
    # (Pdb) p consensus_knowledge.view(-1, 1).shape
    # torch.Size([128, 1])
    # (Pdb) p consensus_knowledge.shape
    # torch.Size([128, 10])
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask
# 它的输入包括三个参数：
#
# knowledge_list: 一个包含了多个源域模型输出的张量，形状为 (batch_size, num_source_domains, num_classes)，其中 batch_size 是一个 batch 的大小，num_source_domains 是源域的数量，num_classes 是分类数。
#
# confidence_gate: 一个浮点数，用于判断每个样本的置信度是否达到门限，超过门限则认为该样本可信。
#
# num_classes: 一个整数，代表分类数。
#
# 函数的输出包括三个张量：
#
# consensus_knowledge_conf: 一个形状为 (batch_size,) 的张量，代表每个样本的知识一致性的置信度。
#
# consensus_knowledge: 一个形状为 (batch_size, num_classes) 的 one-hot 张量，代表每个样本的知识一致性，即每个样本最终被分类到的类别。
#
# consensus_knowledge_mask: 一个形状为 (batch_size,) 的张量，代表每个样本是否达到了置信门限。


def calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers,
                              num_classes):
    """
    :param consensus_focus_dict: record consensus_focus for each domain
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :param source_domain_numbers: the numbers of source domains
    :param consensus_focus_dict:记录每个域的consensus_focus
: param火炬。张量knowledge_list:记录来自每个源域模型的知识
:参数float confidence . gate:判断使用哪个样本的置信门
:param source_domain_numbers:源域个数
    """
    # knowledge_list: 一个包含了多个源域模型输出的张量
    # (Pdb) p consensus_focus_dict
    # {1: 0, 2: 0, 3: 0, 4: 0}
    domain_contribution = {frozenset(): 0}
    # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
    # (Pdb) p domain_contribution
    # {frozenset(): 0}
    for combination_num in range(1, source_domain_numbers + 1):
        # (Pdb) p source_domain_numbers
        # 4
        combination_list = list(combinations(range(source_domain_numbers), combination_num))
        # (Pdb) combination_list
        # [(0,), (1,), (2,), (3,)]
        # 这表示从0到3的四个数字中选取1个数字的所有可能组合。每个组合都是一个元组，包含一个单独的数字。
        for combination in combination_list:
            consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask = knowledge_vote(
                knowledge_list[:, combination, :], confidence_gate, num_classes)
            domain_contribution[frozenset(combination)] = torch.sum(
                consensus_knowledge_conf * consensus_knowledge_mask).item()
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    # (Pdb) p permutation_list
    # [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)]
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (
                                                            domain_contribution[frozenset(
                                                                permutation[:permutation.index(source_idx) + 1])]
                                                            - domain_contribution[
                                                                frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict

# 这段代码用于计算共识聚焦（consensus focus）。它接受以下参数：
#
# consensus_focus_dict: 一个字典，记录每个域的共识聚焦值。
# knowledge_list: 一个包含了多个源域模型输出的张量，用于记录来自每个源域的知识。
# confidence_gate: 置信门阈值，用于判断哪些样本应该被使用。
# source_domain_numbers: 源域的个数。
# num_classes: 分类任务的类别数。
# 函数首先创建一个空的 domain_contribution 字典，用于记录各个组合对共识聚焦的贡献。然后，通过遍历从1到源域个数的组合数量，生成组合列表 combination_list，其中每个组合都是从0到源域个数的数字中选取一定数量数字的组合。
#
# 接下来，对于每个组合，调用 knowledge_vote() 函数计算共识知识的置信度、知识值和掩码。然后，计算该组合对共识聚焦的贡献，并将其添加到 domain_contribution 字典中。
#
# 之后，通过生成源域索引的排列列表 permutation_list，对于每个排列，计算每个源域的共识聚焦值的增量，并将其累加到 consensus_focus_dict 字典中。
#
# 最后，返回更新后的 consensus_focus_dict 字典。
#
# 总体上，这段代码通过对不同的组合和排列进行计算，得到每个源域的共识聚焦值，以衡量其对于共识知识的贡献程度。
def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    """
    Split one epoch into r rounds and perform model aggregation
    :param communication_rounds: the communication rounds in training process
    :param epoch_samples: the samples for each epoch
    :param batch_size: the batch_size for each epoch
    :param total_epochs: the total epochs for training
    :return: batch_per_epoch, total_epochs with communication rounds r
    将一个epoch分割为r轮并执行模型聚合
    :param communication_rounds:训练过程中的交流轮数
    :param epoch_samples:每个epoch的样本
    :param batch_size:每个epoch的batch_size
    :param total_epochs:训练的总epoch
    :返回:batch_per_epoch, total_epoch与通信轮r
    """


    if communication_rounds >= 1:
        epoch_samples = round(epoch_samples / communication_rounds)
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    elif communication_rounds in [0.2, 0.5]:
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    else:
        raise NotImplementedError(
            "The communication round {} illegal, should be 0.2 or 0.5".format(communication_rounds))
    return batch_per_epoch, total_epochs
# (Pdb) p communication_rounds
# 1---5
# (Pdb) p epoch_samples
# 30000
# (Pdb) p batch_size
# 50
# (Pdb) p total_epochs
# 80

# (Pdb) p batch_per_epoch
# 600 ---120
# (Pdb) p total_epochs
# 80 ---400

