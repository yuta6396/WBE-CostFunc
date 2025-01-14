import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from skopt import gp_minimize
from skopt.space import Real
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from optimize import random_search
from analysis import *
from make_directory import make_directory
from config import coefficient_tanh, coefficient_accumulated, target_val, target_ratio
from calc_object_val import calculate_objective_func_val

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
    
    cost = calculate_objective_func_val(control_input, sum_co, Opt_purpose, num_input_grid)
    return sum_co, sum_no, cost


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
    objective_val = calculate_objective_func_val(control_input, sum_co, Opt_purpose, num_input_grid)

    return objective_val


###実行
make_directory(base_dir)
 
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
f.write(f"\n{coefficient_tanh=}")
f.write(f"\n{coefficient_accumulated=}")
f.write(f"\n{target_ratio=}")
f.write(f"\n{target_val=}")
################
f.close()

BO_ratio_matrix = np.zeros((len(max_iter_vec), trial_num)) # iterの組み合わせ, 試行回数
RS_ratio_matrix = np.zeros((len(max_iter_vec), trial_num))
BO_time_matrix = np.zeros((len(max_iter_vec), trial_num)) 
RS_time_matrix = np.zeros((len(max_iter_vec), trial_num))
BO_cost_matrix = np.zeros((len(max_iter_vec), trial_num))
RS_cost_matrix = np.zeros((len(max_iter_vec), trial_num))

BO_file = os.path.join(base_dir, "summary", f"{Alg_vec[0]}.txt")
RS_file = os.path.join(base_dir, "summary", f"{Alg_vec[1]}.txt")


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
            sum_co, sum_no, BO_cost_matrix[exp_i, trial_i] = sim(min_input)
            SUM_no = sum_no
            BO_ratio_matrix[exp_i, trial_i] = calculate_PREC_rate(sum_co, sum_no)
            BO_time_matrix[exp_i, trial_i] = time_diff


            ###RS
            random_reset(trial_i)
            # パラメータの設定
            bounds_MOMY = [(-max_input, max_input)]*num_input_grid  # 探索範囲
            start = time.time()  # 現在時刻（処理開始前）を取得
            if exp_i == 0:
                best_params, best_score = random_search(black_box_function, bounds_MOMY, random_iter_vec[exp_i], f_RS)
            else:
                np.random.rand(int(cnt_base*num_input_grid)) #同じ乱数列の続きを利用したい
                best_params, best_score = random_search(black_box_function, bounds_MOMY, random_iter_vec[exp_i], f_RS, previous_best=(best_params, best_score))
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start

            f_RS.write(f"\n最小値:{best_score}")
            f_RS.write(f"\n入力値:{best_params}")
            f_RS.write(f"\n経過時間:{time_diff}sec")
            f_RS.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
            sum_co, sum_no, RS_cost_matrix[exp_i, trial_i] = sim(best_params)
            sum_RS_MOMY = sum_co
            RS_ratio_matrix[exp_i, trial_i] =  calculate_PREC_rate(sum_co, sum_no)
            RS_time_matrix[exp_i, trial_i] = time_diff

#シミュレーション結果の可視化
filename = f"summary.txt"
config_file_path = os.path.join(base_dir, "summary", filename)  
f = open(config_file_path, 'w')

vizualize_simulation(BO_ratio_matrix, RS_ratio_matrix, BO_time_matrix, RS_time_matrix, 
        BO_cost_matrix, RS_cost_matrix, max_iter_vec,
        f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)
f.close()
