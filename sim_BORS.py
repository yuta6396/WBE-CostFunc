import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from skopt import gp_minimize
from skopt.space import Real
from datetime import datetime
import random
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

matplotlib.use('Agg')

"""
BORSのシミュレーション
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
max_input = 30 #20240830現在ではMOMY=30, RHOT=10, QV=0.1にしている
Alg_vec = ["BO", "RS"]
num_input_grid = 3 #y=20~20+num_input_grid-1まで制御
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択

initial_design_numdata_vec = [3] #BOのRS回数
max_iter_vec = [10, 20, 20, 50, 50, 50]            #{10, 20, 20, 50]=10, 30, 50, 100と同値
random_iter_vec = max_iter_vec

trial_num = 10  #箱ひげ図作成時の繰り返し回数

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"result/BORS/{current_time}/"
cnt_vec = np.zeros(len(max_iter_vec))
for i in range(len(max_iter_vec)):
    if i == 0:
        cnt_vec[i] = int(max_iter_vec[i])
    else :
        cnt_vec[i] = int(cnt_vec[i-1] + max_iter_vec[i])
"""
gp_minimize で獲得関数を指定: acq_func。
gp_minimize の呼び出しにおける主要なオプションは次の通りです。
"EI": Expected Improvement
"PI": Probability of Improvement
"LCB": Lower Confidence Bound
"gp_hedge": これらの獲得関数をランダムに選択し、探索を行う

EI は、探索と活用のバランスを取りたい場合に多く使用されます。
PI は、最速で最良の解を見つけたい場合に適していますが、早期に探索が止まるリスクがあります。
LCB は、解の探索空間が不確実である場合に有効で、保守的に最適化を進める場合に使用されます
"""




nofpe = 2
fny = 2
fnx = 1
run_time = 20

sim_time_sec = 3600 # WBEは1h行われる

varname = 'PREC'

init_file = "init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"

orgfile = 'no-control.pe######.nc'
file_path = os.path.dirname(os.path.abspath(__file__))
gpyoptfile=f"gpyopt.pe######.nc"

cnt=0

def prepare_files(pe: int):
    """ファイルの準備と初期化を行う"""
    output_file = f"out-{input_var}.pe######.nc"
    # input file
    init = init_file.replace('######', str(pe).zfill(6))
    org = org_file.replace('######', str(pe).zfill(6))
    history = history_file.replace('######', str(pe).zfill(6))
    output = output_file.replace('######', str(pe).zfill(6))
    history_path = file_path+'/'+history
    if (os.path.isfile(history_path)):
        subprocess.run(["rm", history])
    subprocess.run(["cp", org, init])  # 初期化

    return init, output

def update_netcdf(init: str, output: str, pe: int, input_values):
    """NetCDFファイルの変数を更新する"""
    with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
        # グローバル属性のコピー
        dst.setncatts(src.__dict__)
        # 次元のコピー
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # 変数のコピーと更新
        for name, variable in src.variables.items():
            x = dst.createVariable(
                name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
            if name == input_var:
                var = src[name][:]
                if pe == 1:
                    for Ygrid_i in range(num_input_grid):
                        var[Ygrid_i, 0, 0] += input_values[Ygrid_i]  # (y, x, z)
                dst[name][:] = var
            else:
                dst[name][:] = src[name][:]

    # outputをinitにコピー
    subprocess.run(["cp", output, init])
    return init

def sim(control_input):
    """
    制御入力決定後に実際にその入力値でシミュレーションする
    """
    for pe in range(nofpe):
        init, output = prepare_files(pe)
        init = update_netcdf(init, output, pe, control_input)

    subprocess.run(["mpirun", "-n", "2", "./scale-rm", "run_R20kmDX500m.conf"])

    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        onc = netCDF4.Dataset(orgfile.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            odat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]

    sum_co=np.zeros(40) #制御後の累積降水量
    sum_no=np.zeros(40) #制御前の累積降水量
    for i in range(40):
        sum_co[i]+=dat[1,0,i,0]*sim_time_sec
        sum_no[i]+=odat[1,0,i,0]*sim_time_sec
    return sum_co, sum_no

def calculate_objective_func_val(sum_co):
    """
    得られた各地点の累積降水量予測値(各Y-grid)から目的関数の値を導出する
    """
    represent_prec = 0
    if Opt_purpose == "MinSum" or Opt_purpose == "MaxSum":
        represent_prec = np.sum(sum_co)
        print(represent_prec)

    elif Opt_purpose == "MinMax" or Opt_purpose == "MaxMax":
        represent_prec = 0
        for j in range(40):  
            if sum_co[j] > represent_prec:
                represent_prec = sum_co[j] # 最大の累積降水量地点
    else:
        raise ValueError(f"予期しないOpt_purposeの値: {Opt_purpose}")

    if Opt_purpose == "MaxSum" or Opt_purpose == "MaxMax":
        represent_prec = -represent_prec # 目的関数の最小化問題に統一   
    return represent_prec


def black_box_function(control_input):
    """
    制御入力値列を入れると、制御結果となる目的関数値を返す
    """
    for pe in range(nofpe):
        init, output = prepare_files(pe)
        init = update_netcdf(init, output, pe, control_input)

    subprocess.run(["mpirun", "-n", "2", "./scale-rm", "run_R20kmDX500m.conf"])

    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]

        sum_co=np.zeros(40) #制御後の累積降水量
        for i in range(40):
            sum_co[i] += dat[1, 0, i, 0] * sim_time_sec
    result = 0
    #f.write(f"\ncnt={int(cnt + cnt_base)}  :input={control_input}")
    objective_val = calculate_objective_func_val(sum_co)

    return objective_val


#### ブラックボックス最適化手法 ####
###ランダムサーチ アルゴリズム
def random_search(objective_function, bounds, n_iterations, previous_best=None):
    # 以前の最良のスコアとパラメータを初期化
    input_history=[]
    if previous_best is None:
        best_score = float('inf')
        best_params = None
    else:
        best_params, best_score = previous_best
    for _ in range(n_iterations):
        candidate = [np.random.uniform(b[0], b[1]) for b in bounds]
        input_history.append(candidate)
        score = objective_function(candidate)
        if score < best_score:
            best_score = score
            best_params = candidate
    f_RS.write(f"\n input_history \n{input_history}")

    return best_params, best_score


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

def figure_BarPlot(exp_i:int, target_data:str, data):
    """
    箱ひげ図を描画し保存する
    """
    # 箱ひげ図の作成
    fig, ax = plt.subplots(figsize=(5, 5))
    box = ax.boxplot(data, patch_artist=True,medianprops=dict(color='black', linewidth=1))

    # 箱ひげ図の色設定
    for patch, color in zip(box['boxes'], colors6):
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

def figire_LineGraph(BO_vec, RS_vec, central_value:str):
    """
    折れ線グラフを描画し保存する
    """
    lw = 2
    ms = 8
    plt.figure(figsize=(8, 6))
    plt.plot(cnt_vec, BO_vec, marker='o', label=Alg_vec[0], color=colors6[0], lw=lw, ms=ms)
    plt.plot(cnt_vec, RS_vec, marker='D', label=Alg_vec[1], color=colors6[3], lw=lw, ms=ms)
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

def write_summary(BO_vec, RS_vec, central_value:str):
    f.write(f"\n{central_value} Accumulated PREC(%) {trial_num} times; BBF={cnt_vec}")
    f.write(f"\n{BO_vec=}")
    f.write(f"\n{RS_vec=}")

###実行

os.makedirs(base_dir, exist_ok=False)
# 階層構造を作成
sub_dirs = ["Accumlated-PREC-BarPlot", "Time-BarPlot", "Line-Graph", "summary"]
for sub_dir in sub_dirs:
    path = os.path.join(base_dir, sub_dir)
    os.makedirs(path, exist_ok=True) 

 
filename = f"config.txt"
config_file_path = os.path.join(base_dir, filename)  # 修正ポイント
f = open(config_file_path, 'w')
###設定メモ###
f.write(f"input_var ={input_var}")
f.write(f"\nmax_input ={max_input}")
f.write(f"\nAlg_vec ={Alg_vec}")
f.write(f"\nnum_input_grid ={num_input_grid}")
f.write(f"\nOpt_purpose ={Opt_purpose}")
f.write(f"\ninitial_design_numdata_vec = {initial_design_numdata_vec}")
f.write(f"\nmax_iter_vec = {max_iter_vec}")
f.write(f"\nrandom_iter_vec = {random_iter_vec}")
f.write(f"\ntrial_num = {trial_num}")
################
f.close()

BO_file = os.path.join(base_dir, "summary", f"{Alg_vec[0]}.txt")
RS_file = os.path.join(base_dir, "summary", f"{Alg_vec[1]}.txt")

filename = f"summary.txt"
config_file_path = os.path.join(base_dir, "summary", filename)  
f = open(config_file_path, 'w')

BO_ratio_matrix = np.zeros((len(max_iter_vec), trial_num)) # iterの組み合わせ, 試行回数
RS_ratio_matrix = np.zeros((len(max_iter_vec), trial_num))

BO_time_matrix = np.zeros((len(max_iter_vec), trial_num)) # iterの組み合わせ, 試行回数
RS_time_matrix = np.zeros((len(max_iter_vec), trial_num))
with open(BO_file, 'w') as f_BO, open(RS_file, 'w') as f_RS:
    for trial_i in range(trial_num):
        cnt_base = 0
        for exp_i in range(len(max_iter_vec)):
            if exp_i > 0:
                cnt_base  = cnt_vec[exp_i - 1]

            ###BO
            random_reset(trial_i)
        # 入力次元と最小値・最大値の定義
            bounds = []
            
            for i in range(num_input_grid):
                bounds.append(Real(-max_input, max_input, name=f'{input_var}_Y{i+20}'))


            start = time.time()  # 現在時刻（処理開始前）を取得
            # ベイズ最適化の実行
            if exp_i == 0:
                result = gp_minimize(
                    func=black_box_function,        # 最小化する関数
                    dimensions=bounds,              # 探索するパラメータの範囲
                    acq_func="EI",
                    n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                    n_initial_points=initial_design_numdata_vec[exp_i],  # 初期探索点の数
                    verbose=True,                   # 最適化の進行状況を表示
                    initial_point_generator = "random",
                    random_state = trial_i
                )
            else:
                result = gp_minimize(
                    func=black_box_function,        # 最小化する関数
                    dimensions=bounds,              # 探索するパラメータの範囲
                    acq_func="EI",
                    n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                    n_initial_points=0,  # 初期探索点の数
                    verbose=True,                   # 最適化の進行状況を表示
                    initial_point_generator = "random",
                    random_state = trial_i,
                    x0=initial_x_iters,
                    y0=initial_y_iters
                )           
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start
            # 最適解の取得
            min_value = result.fun
            min_input = result.x
            initial_x_iters = result.x_iters
            initial_y_iters = result.func_vals
            f_BO.write(f"\n input\n{result.x_iters}")
            f_BO.write(f"\n output\n {result.func_vals}")
            f_BO.write(f"\n最小値:{min_value}")
            f_BO.write(f"\n入力値:{min_input}")
            f_BO.write(f"\n経過時間:{time_diff}sec")
            f_BO.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
            sum_co, sum_no = sim(min_input)
            SUM_no = sum_no
            BO_ratio_matrix[exp_i, trial_i] = calculate_PREC_rate(sum_co, sum_no)
            BO_time_matrix[exp_i, trial_i] = time_diff


            ###Random
            random_reset(trial_i)
            # パラメータの設定
            bounds_MOMY = [(-max_input, max_input)]*num_input_grid  # 探索範囲
            start = time.time()  # 現在時刻（処理開始前）を取得
            if exp_i == 0:
                best_params, best_score = random_search(black_box_function, bounds_MOMY, random_iter_vec[exp_i])
            else:
                np.random.rand(int(cnt_base*num_input_grid)) #同じ乱数列の続きを利用したい
                best_params, best_score = random_search(black_box_function, bounds_MOMY, random_iter_vec[exp_i] , previous_best=(best_params, best_score))
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start

            f_RS.write(f"\n最小値:{best_score}")
            f_RS.write(f"\n入力値:{best_params}")
            f_RS.write(f"\n経過時間:{time_diff}sec")
            f_RS.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
            sum_co, sum_no = sim(best_params)
            sum_RS_MOMY = sum_co
            RS_ratio_matrix[exp_i, trial_i] =  calculate_PREC_rate(sum_co, sum_no)
            RS_time_matrix[exp_i, trial_i] = time_diff


#累積降水量の箱ひげ図比較
f.write(f"\nBO_ratio_matrix = \n{BO_ratio_matrix}")
f.write(f"\nRS_ratio_matrix = \n{RS_ratio_matrix}")
for exp_i in range(len(max_iter_vec)):
    data = [BO_ratio_matrix[exp_i, :], RS_ratio_matrix[exp_i, :]]
    figure_BarPlot(exp_i, "PREC", data)

#timeの箱ひげ図比較
f.write(f"\nBO_time_matrix = \n{BO_time_matrix}")
f.write(f"\nRS_time_matrix = \n{RS_time_matrix}")
for exp_i in range(len(max_iter_vec)):
    data = [BO_time_matrix[exp_i, :], RS_time_matrix[exp_i, :]]
    figure_BarPlot(exp_i, "Time", data)


    #累積降水量の折れ線グラフ比較
#各手法の平均値比較
BO_vec = np.zeros(len(max_iter_vec)) #各要素には各BBFの累積降水量%平均値が入る
RS_vec = np.zeros(len(max_iter_vec))  

for exp_i in range(len(max_iter_vec)):
    BO_vec[exp_i] = np.mean(BO_ratio_matrix[exp_i, :])
    RS_vec[exp_i] = np.mean(RS_ratio_matrix[exp_i, :])

write_summary(BO_vec, RS_vec, "Mean")
figire_LineGraph(BO_vec, RS_vec, "Mean")

#各手法の中央値比較

for exp_i in range(len(max_iter_vec)):
    BO_vec[exp_i] = np.median(BO_ratio_matrix[exp_i, :])
    RS_vec[exp_i] = np.median(RS_ratio_matrix[exp_i, :])

write_summary(BO_vec, RS_vec, "Median")
figire_LineGraph(BO_vec, RS_vec, "Median")

#各手法の最小値(good)比較
for trial_i in range(trial_num):
    for exp_i in range(len(max_iter_vec)):
        if trial_i == 0:
            BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
            RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

        if BO_ratio_matrix[exp_i, trial_i] < BO_vec[exp_i]:
            BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
        if RS_ratio_matrix[exp_i, trial_i] < RS_vec[exp_i]:
            RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

write_summary(BO_vec, RS_vec, "Min")
figire_LineGraph(BO_vec, RS_vec, "Min")

#各手法の最大値(bad)比較
for trial_i in range(trial_num):
    for exp_i in range(len(max_iter_vec)):
        if trial_i == 0:
            BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
            RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

        if BO_ratio_matrix[exp_i, trial_i] > BO_vec[exp_i]:
            BO_vec[exp_i] = BO_ratio_matrix[exp_i, trial_i]
        if RS_ratio_matrix[exp_i, trial_i] > RS_vec[exp_i]:
            RS_vec[exp_i] = RS_ratio_matrix[exp_i, trial_i]

write_summary(BO_vec, RS_vec, "Max")
figire_LineGraph(BO_vec, RS_vec, "Max")

f.close()
