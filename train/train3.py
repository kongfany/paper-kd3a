# 增加一个epoch的数据量，而不是性能还没有提升很高就进行了回退。

# 上传个性化模型，源域由副模型接收，目标域由个性化模型接收
import pdb

import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train(train_dloader_list,model_deputy_list, model_list, classifier_deputy_list,classifier_list,optimizer_deputy_list, optimizer_list,
          classifier_optimizer_deputy_list,classifier_optimizer_list,optimizer_scheduler_deputy_list,optimizer_scheduler_list,
          classifier_optimizer_scheduler_deputy_list,classifier_optimizer_scheduler_list,DET_stage_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level,mix_aug=True):

    # (Pdb) p batch_per_epoch
    # 120

    # debugger
    # train_f1 = test_source(args.source_domains,train_dloader_list,model_list,classifier_list)
    # train_f1_deputy=test_source(args.source_domains,train_dloader_list,model_deputy_list,classifiers_deputy_list)
    # if (train_f1_deputy < alpha1 * train_f1) or DET_stage == 0:
    #     DET_stage = 1
    #     # print_cz('recover', f=logfile)
    #     # print_cz('personalized is teacher', f=logfile)
    # elif (train_f1_deputy >= alpha1 * train_f1 and DET_stage == 1) or (DET_stage >= 2 and train_f1_deputy < alpha2 * train_f1):
    #     DET_stage = 2
    #     # print_cz('exchange', f=logfile)
    #     # print_cz('mutual learning', f=logfile)
    # elif train_f1_deputy >= alpha2 * train_f1 and DET_stage >= 2:
    #     DET_stage = 3
    #     print_cz('sublimate', f=logfile)
    #     # print_cz('deputy is teacher', f=logfile)
    # else:
    #     print_cz('***********************Logic error************************', f=logfile)
    #     DET_stage = 4

    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()

    for model in model_deputy_list:
        model.train()
    for classifier in classifier_deputy_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    # #如果通信轮询<1，
    # #那么我们在(1/communication_rounds) epoch之后执行参数聚合
    # #如果通信轮询>=1:
    # 然后我们延长训练周期，每个周期使用更少的样本。
    # 通信轮次？
    pdb.set_trace()
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 5
    #batch_per_epoch_s = batch_per_epoch / model_aggregation_frequency
    for f in range(model_aggregation_frequency):# 5
        current_domain_index = 0
        # Train model locally on source domains在源域上局部训练模型 5个源域

        for train_dloader, model, classifier, optimizer, classifier_optimizer,optimizer_scheduler,classifier_optimizer_scheduler,model_deputy,classifier_deputy,\
            optimizer_deputy,classifier_optimizer_deputy,optimizer_scheduler_deputy,classifier_optimizer_scheduler_deputy,det_stage in zip(train_dloader_list[1:],
                                                                          model_list[1:],
                                                                          classifier_list[1:],
                                                                          optimizer_list[1:],
                                                                          classifier_optimizer_list[1:],
                                                                          optimizer_scheduler_list[1:],
                                                                          classifier_optimizer_scheduler_list[1:],

                                                                          model_deputy_list[1:],
                                                                          classifier_deputy_list[1:],
                                                                          optimizer_deputy_list[1:],
                                                                          classifier_optimizer_deputy_list[1:],
                                                                          optimizer_scheduler_deputy_list[1:],
                                                                          classifier_optimizer_scheduler_deputy_list[1:],

                                                                          DET_stage_list):
            # DET状态判断
            # 准确率？
            # train_acc,train_f1 = test_source(args.source_domains, train_dloader_list, model_list, classifier_list,345)
            # train_acc_deputy,train_f1_deputy = test_source(args.source_domains,train_dloader_list,model_deputy_list,classifier_deputy_list,345)
            # alpha1=0.7
            # alpha2=0.9
            # if (train_f1_deputy < alpha1 * train_f1) or DET_stage == 0:
            #     DET_stage = 1
            # elif (train_f1_deputy >= alpha1 * train_f1 and DET_stage == 1) or (DET_stage >= 2 and train_f1_deputy < alpha2 * train_f1):
            #     DET_stage = 2
            # elif train_f1_deputy >= alpha2 * train_f1 and DET_stage >= 2:
            #     DET_stage = 3
            # else:
            #     DET_stage = 4

            # check if the source domain is the malicious domain with poisoning attack检查源域是否为投毒攻击的恶意域
            source_domain = source_domains[current_domain_index]

            #  DET状态判断
            train_acc, train_f1 = test_source(source_domain, train_dloader, model, classifier, 345)
            train_acc_deputy, train_f1_deputy = test_source(source_domain, train_dloader, model_deputy,
                                                            classifier_deputy, 345)
            print("Before train:::Source Domain {} Accuracy:{:.3f} fl:{:.3f}".format(source_domain,train_acc,train_f1))
            print("Deputy model:::Accuracy:{:.3f} fl:{:.3f}".format(train_acc_deputy, train_f1_deputy))
            alpha1 = 0.7# 0.5 # 0.75
            alpha2 = 0.9# 0.6 #0.95
            if (train_f1_deputy < alpha1 * train_f1) or det_stage == 0:
                det_stage = 1
                # 2/3 增加鲁棒性， 22 33 22 111          epoch 下 第一次判断的时候增加一个周期性的变化。
            elif (train_f1_deputy >= alpha1 * train_f1 and det_stage == 1) or (det_stage >= 2 and train_f1_deputy < alpha2 * train_f1):
                det_stage = 2
            elif train_f1_deputy >= alpha2 * train_f1 and det_stage >= 2:
                det_stage = 3
            else:
                det_stage = 4
            print("Before train ,Source Domain {} DET_stage:{}".format(source_domain, det_stage))

            # after test, set train
            model.train()
            classifier.train()
            model_deputy.train()
            classifier_deputy.train()


            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            #for i, (image_s, label_s) in enumerate(train_dloader):
                # i 表示当前 batch 的索引(721)
                # 每次返回一个包含 batch_size =50个数据和标签的元组 (images, labels)

                # (Pdb) p len(train_dloader)
                # 721
                # (Pdb) p len(image_s)
                # 50
                # (Pdb) p batch_per_epoch
                # 600
                # 重复利用了前120个数据！！！
                # if i >= (batch_per_epoch):
                #     break
            for i, (image_s, label_s) in enumerate(train_dloader):
                #if f * (batch_per_epoch_s) <= i < (f + 1) * (batch_per_epoch_s):
                    if i >= (batch_per_epoch):
                        break
                    image_s = image_s.cuda()
                    label_s = label_s.long().cuda()
                    if poisoning_attack:
                        # perform poison attack on source domain
                        corrupted_num = round(label_s.size(0) * attack_level)
                        # provide fake labels for those corrupted data
                        label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                    # reset grad重置梯度
                    optimizer.zero_grad()
                    classifier_optimizer.zero_grad()

                    optimizer_deputy.zero_grad()
                    classifier_optimizer_deputy.zero_grad()

                    # each source domain do optimize每个源域做优化
                    feature_s = model(image_s)
                    output_s = classifier(feature_s)
                    feature_s_deputy = model_deputy(image_s)
                    output_s_deputy = classifier_deputy(feature_s_deputy)
                    # loss:根据DET状态进行判断
                    # task_loss_s = task_criterion(output_s, label_s)
                    # task_loss_s.backward()
                    # optimizer.step()
                    # classifier_optimizer.step()

                    # 随着迭代次数的增加，a应该逐渐减小（模型的输出精度会随着训练的进行而增加，所以本地模型比重应该增大）
                    # 仅针对个性化模型进行修改
                    lossweighht_a = 1 - (epoch / 40)
                    lossweighht_a = round(lossweighht_a, 3)
                    lossweighht_b = 2 - lossweighht_a

                    if det_stage == 1:
                        # personalized is teacher
                        loss_ce = task_criterion(output_s, label_s)
                        loss = loss_ce

                        loss_deputy_ce = task_criterion(output_s_deputy, label_s)
                        loss_deputy_kl = F.kl_div(F.log_softmax(output_s_deputy, dim=1),
                                                  F.softmax(output_s.clone().detach(), dim=1), reduction='batchmean')# temp
                        # KL损失（Kullback-Leibler Loss）用于度量两个概率分布之间的差异。
                        # KL损失用于衡量较复杂模型（教师模型）的预测分布与较简单模型（学生模型）的预测分布之间的差异，并将教师模型的知识传递给学生模型。
                        # 对辅助输出进行 log_softmax 操作，获得每个类别的对数概率。(学生模型)
                        # 对主要输出进行 softmax 操作，获得每个类别的概率分布。使用 clone().detach() 是为了防止反向传播时对主要输出的梯度传播到辅助输出。(教师模型)

                        # epoch越大，个性化模型性能越好，副模型学习的权重越高
                        loss_deputy =  loss_deputy_ce*lossweighht_a + loss_deputy_kl*lossweighht_b

                    elif det_stage == 2:
                        # mutual learning DET_stage = 2
                        loss_ce = task_criterion(output_s, label_s)
                        loss_kl = F.kl_div(F.log_softmax(output_s, dim=1), F.softmax(output_s_deputy.clone().detach(), dim=1),
                                           reduction='batchmean')
                        loss = loss_ce * lossweighht_a + loss_kl*lossweighht_b

                        loss_deputy_ce = task_criterion(output_s_deputy, label_s)
                        loss_deputy_kl = F.kl_div(F.log_softmax(output_s_deputy, dim=1),
                                                  F.softmax(output_s.clone().detach(), dim=1), reduction='batchmean')
                        loss_deputy = loss_deputy_ce*lossweighht_a + loss_deputy_kl*lossweighht_b

                    elif det_stage == 3:
                        # deputy is teacher
                        loss_ce = task_criterion(output_s, label_s)
                        loss_kl = F.kl_div(F.log_softmax(output_s, dim=1), F.softmax(output_s_deputy.clone().detach(), dim=1),
                                           reduction='batchmean')
                        loss = loss_ce*lossweighht_a + loss_kl*lossweighht_b

                        loss_deputy_ce = task_criterion(output_s_deputy, label_s)
                        loss_deputy = loss_deputy_ce

                    else:
                        # default mutual learning
                        loss_ce = task_criterion(output_s, label_s)
                        loss_kl = F.kl_div(F.log_softmax(output_s, dim=1), F.softmax(output_s_deputy.clone().detach(), dim=1),
                                           reduction='batchmean')
                        loss = loss_ce + loss_kl

                        loss_deputy_ce = task_criterion(output_s_deputy, label_s)
                        loss_deputy_kl = F.kl_div(F.log_softmax(output_s_deputy, dim=1),
                                                  F.softmax(output_s.clone().detach(), dim=1), reduction='batchmean')
                        loss_deputy = loss_deputy_ce + loss_deputy_kl

                    loss.backward()
                    loss_deputy.backward()

                    optimizer.step()
                    classifier_optimizer.step()
                    optimizer_deputy.step()
                    classifier_optimizer_deputy.step()

                    optimizer_scheduler.step()
                    classifier_optimizer_scheduler.step()
                    optimizer_scheduler_deputy.step()
                    classifier_optimizer_scheduler_deputy.step()





    # Domain adaptation on target domain目标域的域自适应   修改利用副模型进行目标域的训练
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    # (Pdb) p consensus_focus_dict
    # {1: 0, 2: 0, 3: 0, 4: 0}
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break

        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        # optimizer_deputy_list[0].zero_grad()
        # classifier_optimizer_deputy_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote知识投票
        with torch.no_grad():
            # 副模型进行知识投票
            # knowledge_list = [torch.softmax(classifier_deputy_list[i](model_deputy_list[i](image_t)), dim=1).unsqueeze(1) for
            #                   i in range(1, source_domain_num + 1)]
            knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1).unsqueeze(1) for
                i in range(1, source_domain_num + 1)]
            knowledge_list = torch.cat(knowledge_list, 1)
        _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
                                                                  num_classes=num_classes)
        # consensus_knowledge 代表每个样本的知识一致性，即每个样本最终被分类到的类别。
        # consensus_weight 代表每个样本是否达到了置信门限
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # (Pdb) p target_weight
        # [93.0, 128]
        # (Pdb) p target_weight
        # [191.0, 256]
        # Perform data augmentation with mixup 使用mixup执行数据增强
        if mix_aug:
            lam = np.random.beta(2, 2)
        else:
            # Do not perform mixup
            lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]

        feature_t = model_list[0](mixed_image)
        output_t = classifier_list[0](feature_t)
        output_t = torch.log_softmax(output_t, dim=1)
        task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()
        optimizer_scheduler_list[0].step()
        classifier_optimizer_scheduler_list.step()

        # feature_t = model_deputy_list[0](mixed_image)
        # output_t = classifier_deputy_list[0](feature_t)
        # output_t = torch.log_softmax(output_t, dim=1)
        # task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
        # task_loss_t.backward()
        # optimizer_deputy_list[0].step()
        # classifier_optimizer_deputy_list[0].step()
        #
        # optimizer_scheduler_deputy_list[0].step()
        # classifier_optimizer_scheduler_deputy_list[0].step()


        # Calculate consensus focus 计算共识焦点
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                         source_domain_num, num_classes)
        # (Pdb) p consensus_focus_dict
        # {1: 55.129620313644395, 2: 23.36125667889913, 3: 49.67688568433127, 4: 1.8338089783986407}
        # {1: 124.01495426893236, 2: 49.256221473217025, 3: 90.40765319267906, 4: 4.364185591538747}
    # Consensus Focus Re-weighting 共识焦点重加权
    target_parameter_alpha = target_weight[0] / target_weight[1]
    # (Pdb) p target_weight
    # [19464.0, 24960]
    # (Pdb) p target_parameter_alpha
    # 0.7798076923076923
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    # (Pdb) p target_weight
    # 0.156
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    # (Pdb) p source_total_weight
    # 0.844
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])

    # (Pdb) p consensus_focus_dict
    # {1: 12898.554105778572, 2: 5319.408883114698, 3: 9029.595406870045, 4: 499.6325756112715}
    # (Pdb) p epoch_domain_weight
    # [12898.554105778572, 5319.408883114698, 9029.595406870045, 499.6325756112715]
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                           epoch_domain_weight]
    # (Pdb) p epoch_domain_weight
    # [0.3923, 0.1618, 0.2747, 0.0152]
    epoch_domain_weight.insert(0, target_weight)
    # (Pdb) p epoch_domain_weight
    # [0.156, 0.3923, 0.1618, 0.2747, 0.0152]
    # Update domain weight with moving average 用移动平均线更新域权重
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)

    # Model aggregation and Batchnorm MMD 模型聚合
    # federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    federated_average(model_list,model_deputy_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # 上传个性化模型，但是接收到副模型上。 目标域接收到个性化模型上
    # Recording domain weight in logs 在日志中记录域权重
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    return domain_weight


def test_origin(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):

    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))

        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)

        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
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

        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))

            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)

            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))

            tmp_score.append(torch.softmax(output_s, dim=1))

            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)

        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score, k=1, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)


        writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
                          global_step=epoch + 1)

        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)
            print("Source Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(domain_s, top_1_accuracy_s,
                                                                              top_5_accuracy_s))
        else:
            print("Source Domain {} Accuracy {:.3f}".format(domain_s, top_1_accuracy_s))



def test_deputy_origin(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):

    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))

        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)

        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)

    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    if top_5_accuracy:
        _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    else:
        _, y_pred = torch.topk(tmp_score, k=1, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)

    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):

        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))

            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)

            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))

            tmp_score.append(torch.softmax(output_s, dim=1))

            tmp_label.append(label_onehot_s)

        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)

        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score, k=1, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)



        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)

            print("Source Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(domain_s, top_1_accuracy_s,
                                                                              top_5_accuracy_s))
        else:
            print("Source Domain {} Accuracy {:.3f}".format(domain_s, top_1_accuracy_s))