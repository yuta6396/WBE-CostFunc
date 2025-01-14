##目的関数の係数
coefficient_tanh = 2
coefficient_accumulated = 50
target_ratio = 0.9 #制御目標
target_val = 96.5034798248 #制御前の結果
#PSO LDWIM
w_max = 0.9
w_min = 0.4

# GA Parameters
bound = 30
gene_length = 3  # Number of genes per individual 制御入力grid数
crossover_rate = 0.8  # Crossover rate
mutation_rate = 0.05  # Mutation rate
lower_bound = -bound  # Lower bound of gene values
upper_bound = bound  # Upper bound of gene values
alpha = 0.5  # BLX-alpha parameter
tournament_size = 3 #選択数なので population以下にする必要あり