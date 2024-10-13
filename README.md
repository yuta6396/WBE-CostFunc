# WBE-CostFunc
## 概要
Warm Bubble Experimentを用いたシミュレーション. 目的関数に累積降水量（原義？）と入力値の大きさを含める(2024/10/13).

初期値介入による制御問題をBO, RS　もしくはPSO, GAの2手法をsimulateできる（ユーザーはsim_OO.pyを実行）.

基本実装：t=0の時の(y, z) = (20, 0), (21, 0), (22, 0)グリッドの変数MOMYを制御する

basicとの違い：目的関数の変更

累積降水量をtanh()に入れて制御目標値を変曲点にすれば制御入力といい感じにバランスとれる
制御なしver の空間全体の累積降水量= 2.412586995620245
式：係数*tanh(制御後PREC-targetPREC)+ InputValue
tanh=[-1, 1]係数いい感じにすればなるべく小さなInputで閾値以下のいい制御ができる

## ファイル構造
- ~/scale-5.5.1/scale-rm/test/tutorial/ideal/WarmBubbleExperiment/WBE-CostFunc
    - sim_BORS.py

    - sim_PSOGA.py

    - optimize.py ブラックボックス最適化手法の実装

    - analysis.py シミュレーション後の処理

    - make_directory ディレクトリ階層構造を作成

    - config.py ブラックボックス最適化手法のハイパーパラメータ

    - calc_object_val.py 目的関数の計算　どんな目的関数にするか！

    - visualize_input.py 入力値の探索過程の可視化（sim_OO.pyには含まれない）

    - results/              # グラフや結果を保存
        - BORS
            - シミュレーション時間ごとのファイル
                - Accumulated-PREC-BarPlot
                - Cost-BarPlot
                - Line-Graph
                - Time-BarPlot
                - summary テキストメモ
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