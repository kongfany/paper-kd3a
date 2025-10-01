import math
import random
import pdb

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
from model.digit5 import CNN, Classifier
from model.amazon import AmazonMLP, AmazonClassifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.domainnet import DomainNet, DomainNetClassifier
from datasets.DigitFive import digit5_dataset_read
from datasets.AmazonReview import amazon_dataset_read
from lib.utils.federated_utils import *
from train.train import train, test
from datasets.MiniDomainNet import get_mini_domainnet_dloader
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
from datasets.DomainNet import get_domainnet_dloader
from datasets.Office31 import get_office31_dloader
import os
from os import path
import shutil
import yaml

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="DigitFive.yaml")
# 修改默认路径
# parser.add_argument('-bp', '--base-path', default="./")
parser.add_argument('-bp', '--base-path', default="/root/autodl-tmp/KD3A")
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
# 会报错BrokenPipeError: [Errno 32] Broken pipe设置默认为0 -j = 0,适当的num_work值可以根据您的硬件和数据集大小进行调整，以最大化性能4
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
# 当命令行中包含 -dp 或 --data-parallel 选项时，参数 data_parallel 的值将被设置为 False。
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args, configs=configs):
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    # 设置数据加载器列表，模型列表，优化器列表，优化器时间表列表
    train_dloaders = []
    test_dloaders = []
    models = []
    classifiers = []
    optimizers = []
    classifier_optimizers = []
    optimizer_schedulers = []
    classifier_optimizer_schedulers = []


    # build dataset
    if configs["DataConfig"]["dataset"] == "DigitFive":
        domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
        # tar=mnist
        # [0]: target dataset, target backbone, [1:-1]: source dataset, source backbone
        # generate dataset for train and target
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = digit5_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"])
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        # generate CNN and Classifier for target domain
        # (Pdb) p args
        # Namespace(base_path='basepath', bn_momentum=0.1, config='DigitFive.yaml', data_parallel=True, gpu='2', momentum=0.9, optimizer='SGD', print_freq=10, source_domains=None, start_epoch=0, target_domain='mnistm', train_time=1, wd=0.0005, workers=16
        models.append(CNN(args.data_parallel).cuda())
        classifiers.append(Classifier(args.data_parallel).cuda())
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        # create DigitFive dataset
        print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, domain,
                                                                            configs["TrainingConfig"]["batch_size"])
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            # generate CNN and Classifier for source domain
            models.append(CNN(args.data_parallel).cuda())
            classifiers.append(Classifier(args.data_parallel).cuda())
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 10
    elif configs["DataConfig"]["dataset"] == "AmazonReview":
        domains = ["books", "dvd", "electronics", "kitchen"]
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = amazon_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"])
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        # generate MLP and Classifier for target domain
        models.append(AmazonMLP(args.data_parallel).cuda())
        classifiers.append(AmazonClassifier(args.data_parallel).cuda())
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        # create DigitFive dataset
        print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, domain,
                                                                            configs["TrainingConfig"]["batch_size"])
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            # generate CNN and Classifier for source domain
            models.append(AmazonMLP(args.data_parallel).cuda())
            classifiers.append(AmazonClassifier(args.data_parallel).cuda())
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 2
    elif configs["DataConfig"]["dataset"] == "OfficeCaltech10":
        domains = ['amazon', 'webcam', 'dslr', "caltech"]
        target_train_dloader, target_test_dloader = get_office_caltech10_dloader(args.base_path,
                                                                                 args.target_domain,
                                                                                 configs["TrainingConfig"]["batch_size"]
                                                                                 , args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        models.append(
            OfficeCaltechNet(configs["ModelConfig"]["backbone"], bn_momentum=args.bn_momentum,
                             pretrained=configs["ModelConfig"]["pretrained"],
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficeCaltechClassifier(configs["ModelConfig"]["backbone"], 10, args.data_parallel).cuda()
        )
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_office_caltech10_dloader(args.base_path, domain,
                                                                                     configs["TrainingConfig"][
                                                                                         "batch_size"], args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            models.append(
                OfficeCaltechNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                 pretrained=configs["ModelConfig"]["pretrained"],
                                 data_parallel=args.data_parallel).cuda())
            classifiers.append(
                OfficeCaltechClassifier(configs["ModelConfig"]["backbone"], 10, args.data_parallel).cuda()
            )
        num_classes = 10
    elif configs["DataConfig"]["dataset"] == "Office31":
        domains = ['amazon', 'webcam', 'dslr']
        target_train_dloader, target_test_dloader = get_office31_dloader(args.base_path,
                                                                         args.target_domain,
                                                                         configs["TrainingConfig"]["batch_size"],
                                                                         args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        models.append(
            OfficeCaltechNet(configs["ModelConfig"]["backbone"], bn_momentum=args.bn_momentum,
                             pretrained=configs["ModelConfig"]["pretrained"],
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficeCaltechClassifier(configs["ModelConfig"]["backbone"], 31, args.data_parallel).cuda()
        )
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_office31_dloader(args.base_path, domain,
                                                                             configs["TrainingConfig"]["batch_size"],
                                                                             args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            models.append(
                OfficeCaltechNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                 pretrained=configs["ModelConfig"]["pretrained"],
                                 data_parallel=args.data_parallel).cuda())
            classifiers.append(
                OfficeCaltechClassifier(configs["ModelConfig"]["backbone"], 31, args.data_parallel).cuda()
            )
        num_classes = 31
    elif configs["DataConfig"]["dataset"] == "MiniDomainNet":
        domains = ['clipart', 'painting', 'real', 'sketch']
        target_train_dloader, target_test_dloader = get_mini_domainnet_dloader(args.base_path, args.target_domain,
                                                                               configs["TrainingConfig"]["batch_size"],
                                                                               args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        models.append(
            DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                      args.data_parallel).cuda())
        classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 126, args.data_parallel).cuda())
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_mini_domainnet_dloader(args.base_path, domain,
                                                                                   configs["TrainingConfig"][
                                                                                       "batch_size"], args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            models.append(DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                    pretrained=configs["ModelConfig"]["pretrained"],
                                    data_parallel=args.data_parallel).cuda())
            classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 126, args.data_parallel).cuda())
        num_classes = 126
    elif configs["DataConfig"]["dataset"] == "DomainNet":

        # 构建数据集
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        #  generate dataset for train and target生成训练和目标数据集
        # print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = get_domainnet_dloader(args.base_path,
                                                                          args.target_domain,
                                                                          configs["TrainingConfig"]["batch_size"],
                                                                          args.workers)
        num = 0
        for i, (images, labels) in enumerate(target_train_dloader):
            num = num+1
        print(num)# 671
        # (Pdb) p len((images, labels))
        # 2
        # (Pdb) p len(images)
        # 50
        # (Pdb) p len(labels)
        # 50
        # (Pdb) p labels
        # tensor([181, 272, 260, 293,  71, 216, 248,  91, 215, 216, 129, 178, 165,  22,
        #          27, 290,  25, 322, 306, 339, 190,  90,  30, 294, 113, 254, 145,  86,
        #         341,  85, 292, 294, 172,  75, 228, 253, 210, 176, 271, 195, 175,  39,
        #         314,  81, 260, 340, 228, 135, 197, 171])

        # (Pdb) p len(target_train_dloader)
        # 671
        # (Pdb) p len(target_test_dloader)
        # 293
        train_dloaders.append(target_train_dloader)
        # (Pdb) p train_dloaders
        # [<torch.utils.data.dataloader.DataLoader object at 0x7fe6481edd00>]
        # (Pdb) p len(train_dloaders)
        # 1
        test_dloaders.append(target_test_dloader)
        # generate model and Classifier for target domain生成目标域的模型和分类器
        models.append(
            DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                      args.data_parallel).cuda())
        #向列表models中添加一个DomainNet模型对象。具体来说，它使用了以下参数：
        # -  configs["ModelConfig"]["backbone"]：指定了使用的主干网络，包括resnet18、resnet50、resnet101等。
        # -  args.bn_momentum：BatchNorm层的动量参数。
        # -  configs["ModelConfig"]["pretrained"]：是否使用预训练的权重。
        # -  args.data_parallel：是否使用数据并行化。如果为True，则使用多GPU并行计算，否则只使用单个GPU。
        #
        # 最终，这个DomainNet对象被移动到GPU上（使用cuda()方法），并添加到了models列表中。
        # 'resnet101' 0.1
        classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())
        #创建一个DomainNetClassifier类的对象，并将该对象添加到名为“classifiers”的列表中。DomainNetClassifier类带有3个参数：backbone（表示使用的网络模型），345（表示类别数），args.data_parallel（表示是否使用数据并行）。最后，这个对象会被移到CUDA设备上以加速其计算。
        domains.remove(args.target_domain)
        args.source_domains = domains
        # (Pdb) p args.target_domain
        # 'clipart'
        # (Pdb) p args.source_domains
        # ['infograph', 'painting', 'quickdraw', 'real', 'sketch']
        # print("target domain {} loaded".format(args.target_domain))

        # create  dataset
        # print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain 源域的数据集
            source_train_dloader, source_test_dloader = get_domainnet_dloader(args.base_path, domain,
                                                                              configs["TrainingConfig"]["batch_size"],
                                                                              args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            # generate model and Classifier for source domain 为源域生成模型和分类器
            models.append(DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                    pretrained=configs["ModelConfig"]["pretrained"],
                                    data_parallel=args.data_parallel).cuda())
            classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())
            # print("Domain {} Preprocess Finished".format(domain))



        num_classes = 345
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
    # federated learning step 1: initialize model with the same parameter (use target as standard)
    # 联邦学习第一步:用相同的参数初始化模型(使用目标作为标准)
    for model in models[1:]:
        for source_weight, target_weight in zip(model.named_parameters(), models[0].named_parameters()):
            # consistent parameters
            source_weight[1].data = target_weight[1].data.clone()


    # create the optimizer for each model 为每个模型创建优化器
    for model in models:# 每个模型，创建一个 SGD 优化器，并将其加入到 optimizers 列表中
        optimizers.append(
            torch.optim.SGD(model.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    for classifier in classifiers:# 每个分类器，并使用 torch.optim.SGD 函数创建一个 SGD 优化器，并将其加入到 classifier_optimizers 列表中
        classifier_optimizers.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))


    # create the optimizer scheduler with cosine annealing schedule
    # 使用余弦退火调度创建优化调度器
    for optimizer in optimizers:
        optimizer_schedulers.append(
            CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    for classifier_optimizer in classifier_optimizers:
        classifier_optimizer_schedulers.append(
            CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))

    # 这段代码是在创建学习率调度器（LR scheduler）用于优化器（optimizer）和分类器优化器（classifier optimizer）的学习率调度。
    #
    # 在深度学习训练过程中，调整学习率是常见的优化技术之一。学习率调度器控制着学习率在训练过程中的变化。
    # 这里使用的是余弦退火学习率调度器（Cosine Annealing LR Scheduler），它可以将学习率随着训练进行逐渐减小，有助于模型更快地收敛并避免过拟合。
    # 具体来说，这段代码使用了一个for循环，遍历优化器列表（optimizers）和分类器优化器列表（classifier_optimizers），并分别使用CosineAnnealingLR函数创建学习率调度器。
    # 这个函数需要三个参数：优化器（optimizer/classifier_optimizer）、训练的总轮数（total_epochs）和最小的学习率（eta_min），其中训练的总轮数和最小学习率都是从训练配置文件（configs）中获取的。
    #
    # 最终，这段代码将创建好的优化器学习率调度器分别添加到optimizer_schedulers和classifier_optimizer_schedulers列表中，供后续训练过程中使用。
    # create the event to save log info创建事件以保存日志信息
    writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs",
                               "train_time:{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "_".join(args.source_domains))
    print("create writer in {}".format(writer_log_dir))
    # (Pdb) p writer_log_dir
    # 'basepath/DomainNet/runs/train_time:1_clipart_infograph_painting_quickdraw_real_sketch'
    if os.path.exists(writer_log_dir):
        flag = input("{} train_time:{} will be removed, input yes to continue:".format(
            configs["DataConfig"]["dataset"], args.train_time))
        if flag == "yes":
            shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)

    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))
    # debugger
    # pdb.set_trace()
    # create the initialized domain weight 创建初始化的域权重
    domain_weight = create_domain_weight(len(args.source_domains))
    # (Pdb) p domain_weight
    # [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
    # adjust training strategy with communication round
    #   batch_size: 50
    #   epoch_samples: 30000  在每个epoch中使用的总数据数
    #   total_epochs: 80
    # communication_rounds: 1--5
    batch_per_epoch, total_epochs = decentralized_training_strategy(
        communication_rounds=configs["UMDAConfig"]["communication_rounds"],
        epoch_samples=configs["TrainingConfig"]["epoch_samples"],
        batch_size=configs["TrainingConfig"]["batch_size"],
        total_epochs=configs["TrainingConfig"]["total_epochs"])
    # (Pdb) p batch_per_epoch
    # 600
    # (Pdb) p total_epochs
    # 80

    # train model
    for epoch in range(args.start_epoch, total_epochs):  # 80

        DET_stages = [0 for i in range(len(args.source_domains))]  # DET status initialization状态初始化

        domain_weight = train(train_dloaders, models, classifiers, optimizers,
                              classifier_optimizers, epoch, writer, num_classes=num_classes,
                              domain_weight=domain_weight, source_domains=args.source_domains,
                              batch_per_epoch=batch_per_epoch, total_epochs=total_epochs,
                              batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
                              communication_rounds=configs["UMDAConfig"]["communication_rounds"],
                              confidence_gate_begin=configs["UMDAConfig"]["confidence_gate_begin"],
                              confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
                              malicious_domain=configs["UMDAConfig"]["malicious"]["attack_domain"],
                              attack_level=configs["UMDAConfig"]["malicious"]["attack_level"],
                              mix_aug=(configs["DataConfig"]["dataset"] != "AmazonReview"))
        # Source Domains:['infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        # Domain Weight :[0.0713, 0.2351, 0.0859, 0.2743, 0.2786]
        # (Pdb) p type(models)
        # <class 'list'>
        # (Pdb) p models[0]==models[1]
        # False
        # (Pdb)

        # 测试模型精度
        test(args.target_domain, args.source_domains, test_dloaders, models, classifiers, epoch,
             writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10))
        for scheduler in optimizer_schedulers:
            scheduler.step(epoch)
        for scheduler in classifier_optimizer_schedulers:
            scheduler.step(epoch)
        # 根据当前的epoch更新优化器和分类器优化器的学习率调度
        # 用于调整其各自优化器的学习率。调用step方法并传入当前epoch数作为其参数，以根据定义的调度更新学习率。
        # 这段代码是用来更新优化器和分类器优化器的学习率调度器的，根据当前的epoch来调整学习率。在深度学习训练中，调整学习率以帮助模型更快地收敛和/或避免局部最小值是很常见的。学习率调度器控制学习率在训练过程中如何变化。
        #
        # 优化器的学习率调度器和分类器优化器的学习率调度器都是通过torch.optim.lr_scheduler类的实例来实现的。
        # 调用step方法并传入当前的epoch数作为参数，可以根据定义的调度表更新学习率。
        # save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            # save target model with epoch, domain, model, optimizer
            save_checkpoint(
                {"epoch": epoch + 1,
                 "domain": args.target_domain,
                 "backbone": models[0].state_dict(),
                 "classifier": classifiers[0].state_dict(),
                 "optimizer": optimizers[0].state_dict(),
                 "classifier_optimizer": classifier_optimizers[0].state_dict()
                 },
                filename="{}.pth.tar".format(args.target_domain))


def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()