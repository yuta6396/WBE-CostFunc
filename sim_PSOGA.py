import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from config import coefficient_tanh, coefficient_accumulated, target_val, target_ratio, w_max, w_min, gene_length, crossover_rate, mutation_rate, lower_bound, upper_bound, alpha, tournament_size
from optimize import *
from analysis import *
from make_directory import make_directory
from calc_object_val import calculate_objective_func_val

matplotlib.use('Agg')

"""
PSOGAのシミュレーション
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
max_input = 30 #20240830現在ではMOMY=30, RHOT=10, QV=0.1にしている
Alg_vec = ["PSO", "GA"]
num_input_grid = 3 #y=20~20+num_input_grid-1まで制御
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択

particles_vec = [5, 10, 10, 10, 10]           # 粒子数
iterations_vec = [2, 3, 5, 10, 15]        # 繰り返し回数
pop_size_vec = particles_vec  # Population size
num_generations_vec = iterations_vec  # Number of generations

# PSO LDWIM
c1 = 2.0
c2 = 2.0

trial_num = 10  # 乱数種の数

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"result/PSOGA/{current_time}/"
cnt_vec = np.zeros(len(particles_vec))
for i in range(len(particles_vec)):
     cnt_vec[i] = int(particles_vec[i])*int(iterations_vec[i])


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
f.write(f"\ninput_var ={input_var}")
f.write(f"\nmax_input ={max_input}")
f.write(f"\nAlg_vec ={Alg_vec}")
f.write(f"\nnum_input_grid ={num_input_grid}")
f.write(f"\nOpt_purpose ={Opt_purpose}")
f.write(f"\nparticles_vec = {particles_vec}")
f.write(f"\niterations_vec = {iterations_vec}")
f.write(f"\npop_size_vec = {pop_size_vec}")
f.write(f"\nnum_generations_vec = {num_generations_vec}")
f.write(f"\ncnt_vec = {cnt_vec}")
f.write(f"\ntrial_num = {trial_num}\n")
f.write(f"w_max={w_max}\n")
f.write(f"w_min={w_min}\n")
f.write(f"c1={c1}\n")
f.write(f"c2={c2}\n")
f.write(f"gene_length={gene_length}\n")
f.write(f"crossover_rate={crossover_rate}\n")
f.write(f"mutation_rate={mutation_rate}\n")
f.write(f"lower_bound={lower_bound}\n")
f.write(f"upper_bound={upper_bound}\n")
f.write(f"alpha={alpha}\n")
f.write(f"tournament_size={tournament_size}\n")
f.write(f"trial_num={trial_num}\n")
f.write(f"{dpi=}")
f.write(f"\n{coefficient_tanh=}")
f.write(f"\n{coefficient_accumulated=}")
f.write(f"\n{target_ratio=}")
f.write(f"\n{target_val=}")
################
f.close()

PSO_ratio_matrix = np.zeros((len(particles_vec), trial_num))
GA_ratio_matrix = np.zeros((len(particles_vec), trial_num))
PSO_time_matrix = np.zeros((len(particles_vec), trial_num))
GA_time_matrix = np.zeros((len(particles_vec), trial_num))
PSO_cost_matrix = np.zeros((len(particles_vec), trial_num))
GA_cost_matrix = np.zeros((len(particles_vec), trial_num))

PSO_file = os.path.join(base_dir, "summary", f"{Alg_vec[0]}.txt")
GA_file = os.path.join(base_dir, "summary", f"{Alg_vec[1]}.txt")

with open(PSO_file, 'w') as f_PSO, open(GA_file, 'w') as f_GA:
    for trial_i in range(trial_num):
        cnt_base = 0
        for exp_i in range(len(particles_vec)):
            if exp_i > 0:
                cnt_base  = cnt_vec[exp_i - 1]


            ###PSO
            random_reset(trial_i)
            # 入力次元と最小値・最大値の定義
            bounds_MOMY = [(-max_input, max_input)]*num_input_grid  # 探索範囲

            start = time.time()  # 現在時刻（処理開始前）を取得
            best_position, result_value  = PSO(black_box_function, bounds_MOMY, particles_vec[exp_i], iterations_vec[exp_i], f_PSO)
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start
            f_PSO.write(f"\n最小値（目的関数）:{result_value[iterations_vec[exp_i]-1]}")
            f_PSO.write(f"\n入力値:{best_position}")
            f_PSO.write(f"\n経過時間:{time_diff}sec")
            f_PSO.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")

            sum_co, sum_no, PSO_cost_matrix[exp_i, trial_i] = sim(best_position)
            PSO_ratio_matrix[exp_i, trial_i] = calculate_PREC_rate(sum_co, sum_no)
            PSO_time_matrix[exp_i, trial_i] = time_diff


            ###GA
            random_reset(trial_i)
            # パラメータの設定
            start = time.time()  # 現在時刻（処理開始前）を取得
            # Run GA with the black_box_function as the fitness function
            best_fitness, best_individual = genetic_algorithm(black_box_function,
                pop_size_vec[exp_i], gene_length, num_generations_vec[exp_i],
                crossover_rate, mutation_rate, lower_bound, upper_bound,
                alpha, tournament_size, f_GA)
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start

            f_GA.write(f"\n最小値（目的関数）:{best_fitness}")
            f_GA.write(f"\n入力値:{best_individual}")
            f_GA.write(f"\n経過時間:{time_diff}sec")
            f_GA.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")

            sum_co, sum_no, GA_cost_matrix[exp_i, trial_i] = sim(best_individual)
            GA_ratio_matrix[exp_i, trial_i] = calculate_PREC_rate(sum_co, sum_no)
            GA_time_matrix[exp_i, trial_i] = time_diff


#シミュレーション結果の可視化
filename = f"summary.txt"
config_file_path = os.path.join(base_dir, "summary", filename)  
f = open(config_file_path, 'w')

vizualize_simulation(PSO_ratio_matrix, GA_ratio_matrix, PSO_time_matrix, GA_time_matrix, 
        PSO_cost_matrix, GA_cost_matrix, particles_vec,
         f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)

f.close()
