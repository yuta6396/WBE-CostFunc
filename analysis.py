import numpy as np  
import random       
import matplotlib.pyplot as plt  
import os           


def random_reset(trial_i:int):
    """  乱数種の準備"""
    np.random.seed(trial_i) #乱数を再現可能にし, seedによる影響を平等にする
    random.seed(trial_i)     # ランダムモジュールのシードも設定
    return

def calculate_PREC_rate(sum_co, sum_no):
    total_sum_no = 0
    total_sum_co = 0
    for i in range(40):
        total_sum_no += sum_no[i]
        total_sum_co += sum_co[i]
    sum_PREC_decrease  = 100*total_sum_co/total_sum_no
    return sum_PREC_decrease


def figure_BarPlot(exp_i:int, target_data:str, data, colors, base_dir, dpi, Alg_vec, cnt_vec):
    """
    箱ひげ図を描画し保存する
    """
    # 箱ひげ図の作成
    fig, ax = plt.subplots(figsize=(5, 5))
    box = ax.boxplot(data, patch_artist=True,medianprops=dict(color='black', linewidth=1))

    # 箱ひげ図の色設定
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # カテゴリラベルの設定
    plt.xticks(ticks=range(1, len(Alg_vec) + 1), labels=Alg_vec, fontsize=18)
    plt.yticks(fontsize=18)

    # グラフのタイトルとラベルの設定
    plt.xlabel('Optimization method', fontsize=18)    
    if target_data == "PREC":
        plt.title(f'1h Accumulated Precipitation (iter = {cnt_vec[exp_i]})', fontsize=16)
        plt.ylabel('Accumulated precipitation (%)', fontsize=18)
    elif target_data == "Time":
        plt.title(f'Diff in calc time between methods (iter = {cnt_vec[exp_i]})', fontsize=18)
        plt.ylabel('Elapsed time (sec)', fontsize=18)
    else:
        raise ValueError(f"予期しないtarget_dataの値: {target_data}")
    plt.grid(True)
    if target_data == "PREC":
        filename = os.path.join(base_dir, "Accumlated-PREC-BarPlot", f"iter={cnt_vec[exp_i]}.png")
    else:
        filename = os.path.join(base_dir, "Time-BarPlot", f"iter={cnt_vec[exp_i]}.png")
    plt.savefig(filename, dpi= dpi)
    plt.close()
    return

def figire_LineGraph(BO_vec, RS_vec, central_value, base_dir, dpi, Alg_vec, color, cnt_vec, trial_num):
    """
    折れ線グラフを描画し保存する
    """
    lw = 2
    ms = 8
    plt.figure(figsize=(8, 6))
    plt.plot(cnt_vec, BO_vec, marker='o', label=Alg_vec[0], color=color[0], lw=lw, ms=ms)
    plt.plot(cnt_vec, RS_vec, marker='D', label=Alg_vec[1], color=color[3], lw=lw, ms=ms)
        # グラフのタイトルとラベルの設定
    plt.title(f'{central_value} value of {trial_num} times', fontsize=20)
    plt.xlabel('Function evaluation times', fontsize=20)
    plt.ylabel('Accumulated Precipitation (%)', fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.grid(True)
    filename = os.path.join(base_dir, "Line-Graph", f"{central_value}-Accumulated-PREC.png")
    plt.savefig(filename, dpi = dpi)
    plt.close()
    return

def write_summary(BO_vec, RS_vec, central_value:str, f, trial_num, cnt_vec):
    f.write(f"\n{central_value} Accumulated PREC(%) {trial_num} times; BBF={cnt_vec}")
    f.write(f"\n{BO_vec=}")
    f.write(f"\n{RS_vec=}")
    return 

def central_summary(BO_vec, RS_vec, central_value:str, f, base_dir, dpi, Alg_vec, color, trial_num:int, cnt_vec):
    figire_LineGraph(BO_vec, RS_vec, central_value, base_dir, dpi, Alg_vec, color, cnt_vec, trial_num)
    write_summary(BO_vec, RS_vec, central_value, f, trial_num, cnt_vec)
    return 

def vizualize_simulation(BO_ratio_matrix, RS_ratio_matrix, BO_time_matrix, RS_time_matrix, max_iter_vec,
         f,  base_dir, dpi, Alg_vec, color, trial_num:int, cnt_vec):
    #累積降水量の箱ひげ図比較
    f.write(f"\nBO_ratio_matrix = \n{BO_ratio_matrix}")
    f.write(f"\nRS_ratio_matrix = \n{RS_ratio_matrix}")
    exp_num = len(max_iter_vec)
    for exp_i in range(exp_num):
        data = [BO_ratio_matrix[exp_i, :], RS_ratio_matrix[exp_i, :]]
        figure_BarPlot(exp_i, "PREC", data, color, base_dir, dpi, Alg_vec, cnt_vec)

    #timeの箱ひげ図比較
    f.write(f"\nBO_time_matrix = \n{BO_time_matrix}")
    f.write(f"\nRS_time_matrix = \n{RS_time_matrix}")
    for exp_i in range(exp_num):
        data = [BO_time_matrix[exp_i, :], RS_time_matrix[exp_i, :]]
        figure_BarPlot(exp_i, "Time", data, color, base_dir, dpi, Alg_vec, cnt_vec)


        #累積降水量の折れ線グラフ比較
    #各手法の平均値比較
    BO_vec = np.zeros(exp_num) #各要素には各BBFの累積降水量%平均値が入る
    RS_vec = np.zeros(exp_num)  

    for exp_i in range(exp_num):
        BO_vec[exp_i] = np.mean(BO_ratio_matrix[exp_i, :])
        RS_vec[exp_i] = np.mean(RS_ratio_matrix[exp_i, :])

    central_summary(BO_vec, RS_vec, "Mean", f, base_dir, dpi, Alg_vec, color, trial_num, cnt_vec)

    #各手法の中央値比較

    for exp_i in range(exp_num):
        BO_vec[exp_i] = np.median(BO_ratio_matrix[exp_i, :])
        RS_vec[exp_i] = np.median(RS_ratio_matrix[exp_i, :])

    central_summary(BO_vec, RS_vec, "Median", f, base_dir, dpi, Alg_vec, color, trial_num, cnt_vec)

    #各手法の最小値(good)比較
    for trial_i in range(trial_num):
        for exp_i in range(exp_num):
            if trial_i == 0:
                BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
                RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

            if BO_ratio_matrix[exp_i, trial_i] < BO_vec[exp_i]:
                BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
            if RS_ratio_matrix[exp_i, trial_i] < RS_vec[exp_i]:
                RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

    central_summary(BO_vec, RS_vec, "Min", f, base_dir, dpi, Alg_vec, color, trial_num, cnt_vec)

    #各手法の最大値(bad)比較
    for trial_i in range(trial_num):
        for exp_i in range(exp_num):
            if trial_i == 0:
                BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
                RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

            if BO_ratio_matrix[exp_i, trial_i] > BO_vec[exp_i]:
                BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
            if RS_ratio_matrix[exp_i, trial_i] > RS_vec[exp_i]:
                RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

    central_summary(BO_vec, RS_vec, "Max", f, base_dir, dpi, Alg_vec, color, trial_num, cnt_vec)
    return