import multiprocessing


def main():
    list = []
    for i in range(72):
        list.append(i)
    # print(list)
    batch_per_epoch = 60
    #batch_per_epoch2=12
    batch_per_epoch2=30

    for f in range(2): #每个epoch 训练5次2
        for i, value in enumerate(list):

            if f*(batch_per_epoch2)<=i<(f+1)*(batch_per_epoch2):
                print(i, value)
        print("---")

    print("CPU核心数为：", multiprocessing.cpu_count())
    # 通常建议将num_workers设置为你计算机CPU核心数的一半到全部，
    # 例如如果你的CPU有8个核心，那么可以将num_workers设置为4到8之间的值。
    # 但是，这也需要考虑其他因素，例如内存大小和数据集大小等。
    # 如果你的计算机内存较小，可以考虑将num_workers设置为较小的值，避免内存不足导致程序崩溃。
    # (base) kfy@GPU1:~$ cat /proc/cpuinfo | grep processor | wc -l
    # 40
    # (base) kfy@GPU1:~$
    # num = 8 加载120个数据--58.20s
    # num = 16 37s
    # num = 32 27s






if __name__ == "__main__":
    main()