
def main():
    target_domain= 'clipart'
    source_domain=['infograph', 'painting', 'quickdraw', 'real', 'sketch']

    print(target_domain+"deputy")
    # print(["deputy"]*4)
    # print(source_domain+["deputy"]*5)
    source_deputy=[item + 'deputy' for item in source_domain]
    print([item + 'deputy' for item in source_domain])

    state =jungle_det(0,0.049,0.045)
    state1 = jungle_det(state,5.455,5.117)
    state2 = jungle_det(state1,10.567,9.324)
    state3 = jungle_det(state2,12.586,11.124)
    state4 = jungle_det(state3,12.983,11.850)


    print(state,state1,state2,state3,state4)
    # det_stage = 0
    # train_f1 = 0.049
    # train_f1_deputy = 0.045
    det_stage = 1
    train_f1 = 5.455
    train_f1_deputy = 5.117
    DET_stages = [0 for i in range(5)]
    for i in range(3):
        testDet(DET_stages)
    print("---",DET_stages)
def testDet(det_list):

    det_list_1 = []
    for i in range(5):
        print(i,det_list)
        for det_stage in det_list:
            stage = jungle_det(det_stage,0.049,0.045)
            det_list_1.append(stage)
        det_list = det_list_1
        det_list_1 =[]
        print(det_list)







def jungle_det(det_stage,train_f1,train_f1_deputy):
    alpha1 = 0.7
    alpha2 = 0.9
    if (train_f1_deputy < alpha1 * train_f1) or det_stage == 0:
        det_stage = 1
    elif (train_f1_deputy >= alpha1 * train_f1 and det_stage == 1) or (
            det_stage >= 2 and train_f1_deputy < alpha2 * train_f1):
        det_stage = 2
    elif train_f1_deputy >= alpha2 * train_f1 and det_stage >= 2:
        det_stage = 3
    else:
        det_stage = 4

    return det_stage

if __name__ == "__main__":
    main()
    import matplotlib.pyplot as plt

    # 定义参与方和模型组件
    participants = ['Participant 1', 'Participant 2', 'Participant 3']
    model_components = ['Model A', 'Model B', 'Model C']

    # 绘制节点和连接
    plt.figure(figsize=(8, 6))
    plt.title('Federated Learning Model Structure')
    plt.axis('off')

    # 绘制参与方节点
    for i, participant in enumerate(participants):
        plt.text(-0.2, i, participant, fontsize=12, ha='right')
        plt.scatter(0, i, color='blue')

    # 绘制模型组件节点
    for i, component in enumerate(model_components):
        plt.text(1.2, i, component, fontsize=12, ha='left')
        plt.scatter(1, i, color='red')

    # 绘制连接线
    for i in range(len(participants)):
        for j in range(len(model_components)):
            plt.plot([0, 1], [i, j], color='gray')

    plt.show()