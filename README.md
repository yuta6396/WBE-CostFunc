# WBE-AutomaticControlUnion-basic
## 概要
Warm Bubble Experimentを用いた, 自動制御連合講演会用のシミュレーション(2024/10/10).

初期値介入による制御問題をBO, PSO, GA, RSの4手法を1度にsimulateできる（ユーザーはmain.pyを実行）.

t=0の時の(y, z) = (20, 0), (21, 0), (22, 0)グリッドの変数MOMYを制御する

## ファイル構造
- ~/scale-5.5.1/scale-rm/test/tutorial/ideal/WarmBubbleExperiment/WBE-AutomaticControlUnion-basic
    - sim_BORS.py

    - sim_PSOGA.py

    - results/              # グラフや結果を保存
        - BORS
            - シミュレーション時間ごとのファイル
                - Accumulated-PREC-BarPlot
                - Line-Graph
                - Time-BarPlot
                - summary
        - PSOGA


## 可能な実験設定
- 制御変数の変更
    - MOMY  (-30~30)
    - RHOT  (-10~10)
    - QV    (-0.1~0.1)

- 制御グリッドの変更
    - y=20~20+num_grid
    - z=0


- 制御目的の変更
    - 観測できる全範囲（y=0~39）の累積降水量の（最小化/最大化）
    - 最大累積降水量地点（y=y'）の累積降水量（最小化/最大化）

## ブラックボックス最適化手法の実装方法（講演会用の設定）
乱数種を10種類用意しシミュレーションを実行。

        np.random.seed(trial_i) 
        random.seed(trial_i) 
     

### ベイズ最適化
Scikit-Optimize ライブラリのgp minimize 関数

### 粒子群最適化
LDIWMを採用（w_max=0.9, w_min=0.4, c1=c2=2.0）

### 遺伝的アルゴリズム
実数値コーディングを採用

選択方式：トーナメント選択（トーナメントサイズ=3）

交叉方法：ブレンド交叉（交叉率=0.8，ブレンド係数α=0.5）

突然変異方法：ランダムな実数値への置き換え（突然変異率=0.05）

### ランダムサーチ
特になし



## 寄稿者

## 参照