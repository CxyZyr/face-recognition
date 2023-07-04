import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import seaborn as sns


def draw_img_for_model():

    # 准备数据
    x = [0.0001,0.0005,0.001,0.005,0.05]
    y1 = [26.548431105047747, 33.133242382901315, 34.91587085038654, 41.91905411550705, 57.14415643474307]
    y2 = [40.07276034561164, 52.97862664847658, 54.50659390632106, 68.39472487494317, 88.94042746703047]
    y3 = [57.68076398362892, 62.32833105957253, 67.35788994997726, 81.68258299226922, 94.16098226466576]

    # 绘制图像

    plt.plot(x, y1, color='blue', label='SDK 2021S4')
    plt.plot(x, y2, color='green', label='Best 2023S2_1')
    plt.plot(x, y3, color='red', label='Best 2023S2_2')

    plt.xscale('log')
    plt.xticks(x,x)

    # 添加对应点和数值
    for i in range(len(x)):
        plt.scatter(x[i], y1[i], color='blue')
        plt.scatter(x[i], y2[i], color='green')
        plt.scatter(x[i], y3[i], color='red')
    #     plt.annotate('{}'.format(int(y1[i])), (x[i],y1[i]))
    #     plt.annotate('{}'.format(int(y2[i])), (x[i],y2[i]))
    #     plt.annotate('{}'.format(int(y3[i])), (x[i],y3[i]))

    # 添加标题和标签
    plt.title('Glasssix Model')
    plt.xlabel('FPR')
    plt.ylabel('TPR')


    # 添加图例
    plt.legend()

    # 显示图像
    plt.show()

def draw_img_for_model2():
    # 准备数据
    x = [1e-4, 5e-4, 1e-3, 5e-3, 5e-2]
    # y1 = [47.58, 59.33, 64.59, 75.36, 88.17]
    # y2 = [81.97, 88.82,91.08,94.68, 97.52]

    # y1 = [49.76 , 64.08 , 70.97 , 85.35 , 97.33]
    # y2 = [69.20 , 83.28 , 87.71 , 95.22 , 99.46]
    # # #
    # y1 = [48.90 , 63.35 , 70.26 , 84.67 , 97.29]
    # y2 = [59.55 , 80.20 , 85.28 , 93.77 , 99.36]
    # #
    # y1 = [74.13 , 81.05 , 83.68 , 92.11 , 96.91]
    # y2 = [77.83 , 95.73 , 96.81 , 97.12 , 97.87]
    #
    y1 = [10.04 , 15.26 , 17.42 , 24.33 , 37.44]
    y2 = [37.54 , 52.74 , 57.59 , 74.07 , 91.72]



    # 绘制图像
    fig, ax = plt.subplots()
    plt.plot(x, y1, color='blue', label='2021S2')
    plt.plot(x, y2, color='green', label='2023S2V1.0.0')

    plt.xscale('log')
    plt.xticks(x, x)

    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))

    # 添加对应点和数值
    for i in range(len(x)):
        plt.scatter(x[i], y1[i], color='blue')
        plt.scatter(x[i], y2[i], color='green')
        plt.annotate('{}'.format(int(y1[i])), (x[i],y1[i]+1))
        plt.annotate('{}'.format(int(y2[i])), (x[i],y2[i]+1))

    # 添加标题和标签
    plt.xlabel('FAR')
    plt.ylabel('TAR')

    # 添加图例
    plt.legend()

    # 显示图像
    plt.savefig('mask_val_1_1.png')
    plt.show()

def draw_img_for_model3():
    # 准备数据
    x = [10,20,30,40,50,60,70,80,90]

    y1 = [1.00 , 1.00 , 1.00 , 1.00 , 0.96 , 0.83 , 0.54 , 0.25 , 0.05 , 1.00 , 1.00 , 1.00 , 1.00 , 0.96 , 0.90 , 0.68 , 0.32 , 0.05]
    y2 = [1.00 , 1.00 , 1.00 , 1.00 , 0.99 , 0.89 , 0.65 , 0.28 , 0.06 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 0.94 , 0.74 , 0.35 , 0.08]
    y3 = [1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 0.99 , 0.95 , 0.67 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 0.99 , 0.95 , 0.67]
    y4 = [1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 0.96 , 0.72 , 0.24 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 0.96 , 0.72 , 0.24]

    y1 = y1[9:]
    y2 = y2[9:]
    y3 = y3[9:]
    y4 = y4[9:]



    # 绘制图像
    plt.figure(figsize=(8, 8))
    plt.plot(x, y1, color='green', label='2021S2',linewidth=0.5)
    plt.plot(x, y2, color='blue', label='2023S2V1.0.0',linewidth=0.5)
    plt.plot(x, y3, color='orange', label='FacePass SDK',linewidth=0.5)
    plt.plot(x, y4, color='red', label='FacePro SDK',linewidth=0.5)

    plt.xticks(x, x)
    plt.ylim(0, 1.1)

    # 添加对应点和数值
    for i in range(len(x)):
        plt.scatter(x[i], y1[i], color='green',s=1)
        plt.scatter(x[i], y2[i], color='blue',s=1)
        plt.scatter(x[i], y3[i], color='orange',s=1)
        plt.scatter(x[i], y4[i], color='red',s=1)
        plt.annotate('{}'.format(float(y1[i])), (x[i],y1[i]+0.01),fontsize=8)
        plt.annotate('{}'.format(float(y2[i])), (x[i],y2[i]+0.01),fontsize=8)
        plt.annotate('{}'.format(float(y3[i])), (x[i], y3[i] + 0.01),fontsize=8)
        plt.annotate('{}'.format(float(y4[i])), (x[i], y4[i] + 0.01),fontsize=8)

    # 添加图例
    plt.legend()

    # 显示图像
    plt.savefig('OFD_angel_right')
    plt.show()

def draw_score():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # 生成示例数据
    scores_path = r'../scores'
    for txt in tqdm(os.listdir(scores_path)):
        scores1 = []
        scores5 = []
        scores10 = []
        txt_path = os.path.join(scores_path, txt)
        save_img_name = txt_path.replace('scores', 'img').replace('.txt', '_1_N.png').replace('0522_backbone_25000.pth','2023S2V1.1.0'.replace('unicorn_pretrain.pth','2021S4'))
        os.makedirs(os.path.dirname(save_img_name), exist_ok=True)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            scores = [float(i) for i in line]
            for i in range(len(scores)):
                if i == 0:
                    scores1.append(scores[i])
                if i < 5:
                    scores5.append(scores[i])
                scores10.append(scores[i])
        # 计算直方图的数据
        bins = np.linspace(0, 1, 101)
        hist1, _ = np.histogram(np.array(scores1), bins=bins)
        hist2, _ = np.histogram(np.array(scores5), bins=bins)
        hist3, _ = np.histogram(np.array(scores10), bins=bins)

        # 创建图形和子图
        fig, ax = plt.subplots(figsize=(16, 8))

        # 设置每个直方图的宽度
        width = (bins[1] - bins[0]) * 0.2

        # 计算每个直方图的位置
        positions = np.arange(0,1,0.01)

        # 绘制直方图
        ax.bar(positions - 0.004, hist1, width=width, color='red', alpha=1, label='top1 distribution')
        ax.bar(positions, hist2, width=width, color='blue', alpha=1, label='top5 distribution')
        ax.bar(positions + 0.004, hist3, width=width, color='#8ECFC9', alpha=1, label='top10 distribution')

        # 设置x轴刻度范围和刻度间隔
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x', rotation=90)
        # 添加图例
        ax.legend()

        # 显示图形
        # plt.show()
        plt.savefig(save_img_name)
        plt.close()

if __name__ == '__main__':
    draw_score()