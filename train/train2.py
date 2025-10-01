import pdb

import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter


def train(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, mix_aug=True):




    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    # 通信轮次？
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    # (Pdb) p batch_per_epoch
    # 600
    # (Pdb) p type(batch_per_epoch)
    # <class 'int'>
    batch_per_epoch = batch_per_epoch/5
    # (Pdb) p batch_per_epoch
    # 120.0
    # (Pdb) p type(batch_per_epoch)
    # <class 'float'>

    for f in range(model_aggregation_frequency):  # 1
        current_domain_index = 0
        # Train model locally on source domains在源域上局部训练模型 5个源域
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                     model_list[1:],
                                                                                     classifier_list[1:],
                                                                                     optimizer_list[1:],
                                                                                     classifier_optimizer_list[1:]):
            for i, (image_s, label_s) in enumerate(train_dloader):
                if f * (batch_per_epoch) <= i < (f + 1) * (batch_per_epoch):
                    print(i)
            print("----")

    for f in range(model_aggregation_frequency):  # 1
        current_domain_index = 0
        # Train model locally on source domains在源域上局部训练模型 5个源域
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                     model_list[1:],
                                                                                     classifier_list[1:],
                                                                                     optimizer_list[1:],
                                                                                     classifier_optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack检查源域是否为投毒攻击的恶意域
            source_domain = source_domains[current_domain_index]
            # (Pdb) p source_domain
            # 'infograph'
            # (Pdb) p train_dloader
            # <torch.utils.data.dataloader.DataLoader object at 0x7f4880551d90>
            # (Pdb) p classifier
            # DomainNetClassifier(
            #   (linear): DataParallel(
            #     (module): Sequential(
            #       (fc): Linear(in_features=2048, out_features=345, bias=True)
            #     )
            #   )
            # )
            # (Pdb) p train_dloader
            # <torch.utils.data.dataloader.DataLoader object at 0x7f4880551d90>



            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break

                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                # (Pdb) p image_s.shape
                # torch.Size([50, 3, 224, 224])
                # (Pdb) p label_s.shape
                # torch.Size([50])
                # (Pdb) p model(image_s).shape
                # torch.Size([50, 2048, 1, 1])
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad重置梯度
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize每个源域做优化
                feature_s = model(image_s)
                output_s = classifier(feature_s)
                # (Pdb) p feature_s.shape
                # torch.Size([50, 2048, 1, 1])
                # (Pdb) p output_s.shape
                # torch.Size([50, 345])
                task_loss_s = task_criterion(output_s, label_s)
                task_loss_s.backward()
                optimizer.step()
                classifier_optimizer.step()


    # Domain adaptation on target domain目标域的域自适应
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # (Pdb) p confidence_gate
    # 0.8
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        # (Pdb) p len(train_dloader_list[0])
        # 671

        # (Pdb) p len(train_dloader_list[0])
        # 196
        # (Pdb) p image_t.shape
        # torch.Size([128, 3, 32, 32])
        # (Pdb) p label_t.shape
        # torch.Size([128]) batch_Size =128
        if i >= batch_per_epoch:
            break
        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote知识投票
        with torch.no_grad():
            knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1).unsqueeze(1) for
                              i in range(1, source_domain_num + 1)]
            # 对于给定的一批图像数据 image_t，对所有的源域模型（共 source_domain_num 个）进行前向传播，
            # 将每个模型的输出（即分类器的输出）进行softmax操作，并在第二个维度上增加一维（变为2维），
            # 然后将所有模型的输出组合成一个列表 knowledge_list。
            # knowledge_list 是一个列表，其中每个元素是一个2维张量，表示对应源域模型对 image_t 的分类结果。
            # (Pdb) p len(knowledge_list)
            # 4
            # (Pdb) p knowledge_list[0].shape
            # torch.Size([128, 1, 10])
            knowledge_list = torch.cat(knowledge_list, 1)
            # (Pdb) p len(knowledge_list)
            # 128
            # (Pdb) p knowledge_list[0].shape
            # torch.Size([4, 10])
            # knowledge_list 中所有张量按照第二个维度（即 dim=1）进行拼接，生成一个新的张量
        _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
                                                                  num_classes=num_classes)
        #
        # consensus_knowledge: 一个形状为 (batch_size, num_classes) 的 one-hot 张量，代表每个样本的知识一致性，即每个样本最终被分类到的类别。
        # 目标域的标签。
        # # consensus_weight: 一个形状为 (batch_size,) 的张量，代表每个样本是否达到了置信门限。
        # (Pdb) p confidence_gate
        # 0.9
        # (Pdb) p num_classes
        # 10
        # 最后，使用knowledge_vote()函数对这些概率分布进行投票，得到整个模型的输出概率分布。
        # 其中，confidence_gate参数用于过滤掉低置信度的预测结果，num_classes参数指定了分类的类别数目。
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # 第一行代码将投票后的权重值加入到目标领域样本数量的计数器中，
        # 第二行代码将投票后的类别数量加入到目标领域类别数量的计数器中。
        # 这个过程是为了计算目标领域在联邦学习中的权重，以便在训练过程中对目标领域的贡献进行加权。
        # (Pdb) p target_weight
        # [25.0, 50]
        # (Pdb)
        # (Pdb) p target_weight
        # [127.0, 128]
        # Perform data augmentation with mixup 使用mixup执行数据增强
        if mix_aug:
            lam = np.random.beta(2, 2)
        else:
            # Do not perform mixup
            lam = np.random.beta(2, 2)
        # (Pdb) p lam
        # 0.827280711292405
        batch_size = image_t.size(0)# 128
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        # (Pdb) p mixed_consensus
        # tensor([[0.0000, 0.8215, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.8215, 0.0000],
        #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.8215, 0.1785],
        #         ...,
        #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.8215, 0.0000],
        #         [0.0000, 0.0000, 0.0000,  ..., 0.1785, 0.0000, 0.8215],
        #         [0.1785, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],
        #        device='cuda:0')

        # 这段代码实现了Mixup数据增强
        # 从Beta分布中随机采样一个lam值；
        # 获取当前batch的大小，即图片数量；
        # 随机生成一个大小为batch_size的索引数组；
        # 计算Mixup后的图片和知识，即将当前图片和知识按照lam和1-lam的比例进行加权平均，得到混合的图片和知识。
        feature_t = model_list[0](mixed_image)
        output_t = classifier_list[0](feature_t)
        # tensor([[-0.5296,  0.2284, -0.0692,  ...,  0.8337, -0.6087, -0.3021],
        #         [-0.3452, -0.0427,  0.3444,  ...,  0.1086, -0.2545, -0.3309],
        #         [-0.6291,  0.0307, -0.1664,  ...,  0.5273, -0.2408, -0.5093],
        #         ...,
        #         [-0.3714, -0.0127, -0.3011,  ...,  0.2770, -0.0453,  0.3194],
        #         [-0.6563, -0.0433,  0.6194,  ..., -0.0898, -0.2026, -0.3774],
        #         [-0.8058, -0.0048, -0.3885,  ...,  0.2709, -0.0333, -0.0591]],
        #        device='cuda:0', grad_fn=<AddmmBackward>)
        output_t = torch.log_softmax(output_t, dim=1)# 函数对输出进行操作，使得输出变为对数概率值
        # (Pdb) p output_t
        # tensor([[-2.9099, -2.1520, -2.4496,  ..., -1.5467, -2.9891, -2.6825],
        #         [-2.6999, -2.3973, -2.0102,  ..., -2.2460, -2.6091, -2.6855],
        #         [-2.8149, -2.1551, -2.3522,  ..., -1.6585, -2.4267, -2.6951],
        #         ...,
        #         [-2.6019, -2.2433, -2.5316,  ..., -1.9536, -2.2758, -1.9112],
        #         [-3.0065, -2.3935, -1.7308,  ..., -2.4400, -2.5528, -2.7276],
        #         [-3.3311, -2.5300, -2.9137,  ..., -2.2543, -2.5585, -2.5843]],
        #        device='cuda:0', grad_fn=<LogSoftmaxBackward>)
        # (Pdb) p output_t.shape
        # torch.Size([128, 10])
        task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))# 计算交叉熵损失函数
        # consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1):
        # 将一致性权重与一致性损失相乘，对每个样本进行加权。 对每个样本计算一致性特征与模型输出的乘积的和。

        # 这里使用的损失函数是交叉熵损失函数。
        # 具体地，首先将混合的一致性标签 mixed_consensus 与目标域的输出 output_t 相乘，
        # 然后对每个样本在类别维度上求和，再乘以一致性权重 consensus_weight。
        # 最后，对所有样本的损失进行平均，得到任务损失 task_loss_t。
        # mixed_consensus: 混合的一致性标签，用于衡量样本在不同域之间的一致性。它是一个与样本数量相同的张量。
        # output_t: 目标域模型的输出，经过 log softmax 转换后的概率分布。它是一个大小为 (batch_size, num_classes) 的张量，其中 batch_size 是批次中的样本数量，num_classes 是分类的类别数。
        # torch.sum(-1 * mixed_consensus * output_t, dim=1): 将混合一致性标签 mixed_consensus 与目标域模型的输出 output_t 相乘，并在类别维度上求和。这一步得到的是每个样本的一致性损失。
        # consensus_weight: 一致性权重，用于加权不同样本的一致性损失。它是一个与样本数量相同的张量。
        # consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1): 将一致性权重与一致性损失相乘，对每个样本进行加权。
        # torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1)): 对所有样本的加权一致性损失进行平均，得到任务损失 task_loss_t。
        # 综上所述，该代码段通过将一致性标签与目标域模型的输出相乘，并加权求和，计算了任务损失。这种损失衡量了目标域中样本的分类一致性，以帮助域适应算法在目标域上进行模型训练。

        # (Pdb) p task_loss_t
        # tensor(2.3722, device='cuda:0', grad_fn=<MeanBackward0>)

        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()
        # Calculate consensus focus 计算共识焦点
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                         source_domain_num, num_classes)

    # Consensus Focus Re-weighting 共识焦点重加权
    target_parameter_alpha = target_weight[0] / target_weight[1]
    # (Pdb) p target_parameter_alpha
    # 0.32303333333333334
    # (Pdb) p target_weight[0]
    # 9691.0
    # (Pdb) p target_weight[1]
    # 30000
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    # 目标域权重是源域权重和目标域权重的加权平均值。这里的目标域权重是由参数target_parameter_alpha控制的。
    # 具体地说，算法首先将target_parameter_alpha平均分配到源域数量+1个域中（包括目标域本身），6
    # 然后将结果四舍五入到小数点后四位，得到每个域的权重。这里的权重指的是每个域在计算加权平均时所占的比例。
    # (Pdb) p target_weight
    # 0.0538
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    # (Pdb) p source_total_weight
    # 0.9462
    for i in range(1, source_domain_num + 1):# 1,6
        epoch_domain_weight.append(consensus_focus_dict[i])
    # (Pdb) p epoch_domain_weight
    # [808.1229669104141, 3198.418831244101, 1136.2885559549513, 3449.2407718053605, 3631.997759888752]
    # (Pdb) p consensus_focus_dict
    # {1: 808.1229669104141, 2: 3198.418831244101, 3: 1136.2885559549513, 4: 3449.2407718053605, 5: 3631.997759888752}
    # (Pdb)
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                           epoch_domain_weight]
    # (Pdb) p epoch_domain_weight
    # [0.0621, 0.234, 0.0837, 0.2748, 0.2934]
    epoch_domain_weight.insert(0, target_weight)
    # (Pdb) p epoch_domain_weight
    # [0.052, 0.0621, 0.234, 0.0837, 0.2748, 0.2934]
    # (Pdb) p epoch_domain_weight
    # [0.0538, 0.0626, 0.2476, 0.088, 0.267, 0.2811]
    # 将每个元素v乘以source_total_weight后除以epoch_domain_weight的总和，以确保各元素之和为source_total_weight
    # 将计算好的权重插入到列表的开头，以表示目标域的权重。
    # Update domain weight with moving average 用移动平均线更新域权重
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    # (Pdb) p type(model_list[0])
    # <class 'model.domainnet.DomainNet'>
    # (Pdb) p type(model_list)
    # <class 'list'>
    # (Pdb) p len(model_list)
    # 6
    # (Pdb) p (model in model_list)
    # True
    # (Pdb) p (model_list.index(model))
    # 5
    # (Pdb) p model_list[5]==model
    # True
    # Model aggregation and Batchnorm MMD 模型聚合

    federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # 聚合后模型并不一致？
    # (Pdb) p model_list[0]==model_list[5]
    # False
    # (Pdb) p model in model_list
    # True
    # (Pdb) p model == model_list[5]
    # True
    # (Pdb) p model_list[1] == model_list[2]
    # False

    # Recording domain weight in logs 在日志中记录域权重
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        # (Pdb) p len(train_dloader_list) - 1
        # 5
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    return domain_weight
    # Source Domains:['infograph', 'painting', 'quickdraw', 'real', 'sketch'], Domain Weight :[0.0713, 0.2351, 0.0859, 0.2743, 0.2786]
    # (Pdb) p domain_weight
    # [0.0549, 0.0713, 0.2351, 0.0859, 0.2743, 0.2786]


def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):
    # (Pdb) p target_domain
    # 'clipart'
    # (Pdb) p source_domains
    # ['infograph', 'painting', 'quickdraw', 'real', 'sketch']
    # (Pdb)
    # (Pdb) p len(model_list)
    # 6
    # (Pdb) p num_classes
    # 345
    # (Pdb) p top_5_accuracy
    # True
    source_domain_losses = [AverageMeter() for i in source_domains]
    # 这里的AverageMeter()指的是一个类，用来测量数据的平均值和当前值。
    # source_domain_losses是一个列表，其中的每个元素都是一个AverageMeter()对象，对应每个源域的损失。
    # 在训练过程中，每个源域的损失会被计算并添加到对应的AverageMeter()对象中，以便在记录日志时能够方便地查看每个源域的损失情况。
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda() #交叉熵损失函数
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # 将`classifier_list`列表中的分类器模型都设置为评估模式（evaluation  mode）。在评估模式下，模型不会更新梯度，而是根据输入数据给出输出
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        # (Pdb) p label_t
        # tensor([ 86, 315,  22, 331, 256, 149, 320, 274, 136, 165,  31, 195, 269, 176,
        #          79,  64, 323, 125,  77, 194, 291,  31,  81, 256, 339, 120, 301, 286,
        #          94,  69,  86, 120, 246,  27,  58, 117, 336, 197, 229, 317,  78, 329,
        #         113,  39,  29, 254, 285, 155, 270, 330])
        # (Pdb) p len(label_t)
        # 50
        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))
            # 这段代码意味着将一个输入的图像(image_t)通过模型(model_list[0])进行处理，
            # 然后通过分类器(classifier_list[0])进行分类，输出结果保存在output_t中。其中使用了列表的索引来获取模型和分类器。
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        # 它将原始的类别标签`label_t`转换为one-hot编码形式，将每个样本的类别标签值放在对应的列上，从而得到一个按照类别进行编码的标签向量。
        # 这个操作可以方便地用于计算交叉熵损失，通常用于神经网络分类任务中。
        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    # 预测值  tmp_score 真值  tmp_label
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    # 预测值  tmp_score 真值  tmp_label

    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    if top_5_accuracy:
        _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    else:
        _, y_pred = torch.topk(tmp_score, k=1, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/target_domain_{}_accuracy_top1".format(target_domain).format(target_domain),
                      scalar_value=top_1_accuracy_t,
                      global_step=epoch + 1)
    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/target_domain_{}_accuracy_top5".format(target_domain).format(target_domain),
                          scalar_value=top_5_accuracy_t,
                          global_step=epoch + 1)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        # (Pdb) p source_domains
        # ['infograph', 'painting', 'quickdraw', 'real', 'sketch']
        # (Pdb) p s_i
        # 0
        # (Pdb) p domain_s
        # 'infograph'
        # (Pdb)
        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))
            # 计算当前图像批次的真实标签的one-hot编码
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)
            #   output_t：模型针对特定任务预测的结果。
            #   label_t：真实标签，即特定任务的正确答案。
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            # 将当前图像批次的模型的softmax输出附加到tmp_score列表中。
            tmp_score.append(torch.softmax(output_s, dim=1))
            # turn label into one-hot code将当前图像批次的真实标签的one-hot编码附加到tmp_label列表中。
            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        # 使用 writer.add_scalar() 函数记录当前测试中源域数据的平均损失，方便在 TensorBoard 中查看。
        # 使用 torch.cat() 函数将所有任务的预测结果和真实标签按照行的维度拼接成一个矩阵。
        #(Pdb) p tmp_score.size(0)
        #15582
        # 使用 torch.topk() 函数获取每个样本预测结果中最大的值和相应的索引，其中参数 k=1 表示只获取最大值和相应的索引。
        # 将索引作为真实标签，得到模型在当前测试中对所有任务的预测标签
        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score, k=1, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)#默认计算的为top5

        # y_true是真实标签的一维向量，y_pred是模型对于每个样本预测的类别的概率分布，取最大概率对应的类别作为预测结果。
        # 代码中通过torch.sum函数计算了预测正确的样本数量，然后除以总样本数求得top-1准确率，并将其赋值给top_1_accuracy_s变量。
        # 最后，top_1_accuracy_s表示当前任务的top-1准确率。
        # (Pdb) p top_1_accuracy_s
        # 0.14176614041843152
        # (Pdb) p domain_s
        # 'infograph'
        # (Pdb)
        # (Pdb) p y_true
        # tensor([[ 77],
        #         [150],
        #         [191],
        #         ...,
        #         [112],
        #         [216],
        #         [131]], device='cuda:0')
        # (Pdb) p y_pred
        # tensor([[ 33, 285, 301, 304, 185],
        #         [301, 278, 312,  33, 325],
        #         [301, 305, 311, 318, 343],
        #         ...,
        #         [301, 278, 161,  33, 328],
        #         [ 62, 216, 334,  10, 290],
        #         [ 43, 127, 284,  63,  10]], device='cuda:0')
        # (Pdb)
        # (Pdb) p y_pred[:, :1]
        # tensor([[ 33],
        #         [301],
        #         [301],
        #         ...,
        #         [301],
        #         [ 62],
        #         [ 43]], device='cuda:0')
        # (Pdb) p torch.sum(y_true == y_pred[:, :1])
        # tensor(2209, device='cuda:0')
        # (Pdb) p torch.sum(y_true == y_pred)
        # tensor(4673, device='cuda:0')

        # (Pdb) p y_true.view(-1)
        # tensor([ 77, 150, 191,  ..., 112, 216, 131], device='cuda:0')
        #(Pdb) p y_pred[:, :1].view(-1)
        #tensor([ 33, 301, 301,  ..., 301,  62,  43], device='cuda:0')
        # (Pdb) p y_true.view(-1) ==  y_pred[:, :1].view(-1)
        # tensor([False, False, False,  ..., False, False, False], device='cuda:0')
        # (Pdb) p torch.sum(y_true.view(-1) ==  y_pred[:, :1].view(-1))
        # tensor(2209, device='cuda:0')

        writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
                          global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)


