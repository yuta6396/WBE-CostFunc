import numpy as np
from config import coefficient_tanh, coefficient_accumulated, target_val, target_ratio


def calculate_objective_func_val(control_input, sum_co, Opt_purpose:str, num_input_grid):
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

    cost = coefficient_accumulated*np.tanh(coefficient_tanh*(represent_prec - target_val*target_ratio))
    if Opt_purpose == "MaxSum" or Opt_purpose == "MaxMax":
        cost = -cost # 目的関数の最小化問題に統一   

    for j in range(num_input_grid):
        cost += abs(control_input[j])
    return cost