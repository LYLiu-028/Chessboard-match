# Chessboard Match

一个基于 OpenCV 的棋盘格匹配小工具。它读取左右相机各自的两张图像，恢复固定规格棋盘格的网格点，并输出两帧之间可用于后续 PnP / 对应点分析的坐标文件，同时生成可视化结果。

## 项目作用

当前脚本会处理两组数据：

- `left/2026020501_qmz.5.Camera_left.png` 与 `left/2026020501_qmz.9.Camera_left.png`
- `right/2026020501_qmz.5.Camera.right.png` 与 `right/2026020501_qmz.9.Camera.right.png`

运行后会输出：

- `pnp_pointsAB_left.txt`
- `pnp_pointsAB_right.txt`
- `match_vis_left.png`
- `match_vis_right.png`

其中 `pnp_pointsAB_*.txt` 每一行格式为：

```text
x_a y_a x_b y_b
```

表示同一个棋盘格单元右下角顶点在图像 A 与图像 B 中的位置；如果某一侧缺失，则写成 `-1 -1`。

## 运行环境

```bash
python3 -m pip install -r requirements.txt
python3 match_checker_cells.py
```

依赖很简单：

- `opencv-python`
- `numpy`

## 实现原理

核心代码集中在 [match_checker_cells.py](./match_checker_cells.py)。

### 1. 用多种预处理策略恢复参考图像的完整内角点

`pick_best_find_chessboard()` 会依次尝试 `gray`、`clahe`、`equalize`、缩放后再增强等多种预处理版本，然后调用 `cv2.findChessboardCorners`。  
这样做的原因是同一块棋盘在不同曝光、模糊和对比度条件下，单一预处理并不稳定，多策略试探可以提高一次命中完整角点网格的概率。

对应逻辑：

- `preprocess_chessboard_variant()`
- `run_find_chessboard_variant()`
- `pick_best_find_chessboard()`

### 2. 将 OpenCV 返回的内角点顺序规整成稳定网格

OpenCV 找到的是内角点，项目中棋盘规格固定为 `47 x 28` 个格子，因此真实需要的是：

- 内角点网格：`46 x 27`
- 外层节点网格：`48 x 29`

`normalize_inner_corner_order()` 负责把角点顺序统一到从上到下、从左到右的布局；  
`extrapolate_full_grid()` 再根据最外一圈内角点做线性外推，把外边框节点补出来。

这一段逻辑的意义是：后续输出的是“格点”而不是仅有的内角点，所以必须从检测到的内角点恢复完整棋盘节点。

### 3. 第二张图优先直接检测，失败时退化为光流跟踪

`transfer_nodes_from_reference()` 是这个项目里最关键的鲁棒性补偿逻辑：

- 先尝试在目标图上再次完整检测棋盘；
- 如果检测不到完整网格，就把参考图上的内角点用 `cv2.calcOpticalFlowPyrLK` 跟踪到目标图；
- 再做一次前后向一致性检查，筛掉漂移点；
- 最后用 `cv2.cornerSubPix` 把有效点做亚像素精修。

这部分的实现原理是：

- 完整检测成功时，结果最可靠，直接采用；
- 完整检测失败时，棋盘局部纹理通常仍然存在，因此 LK 光流可以借助参考帧把网格“搬运”过去；
- 前后向检查保证跟踪不是单向误配；
- 亚像素优化保证输出点能继续用于几何计算，而不是停留在粗像素级。

### 4. 按棋盘单元输出对应关系

`iter_cell_records()` 逐个遍历棋盘单元。  
每个单元取右下角顶点作为导出点，并同时计算 A/B 两张图里这个单元是否有效。

`write_points_file()` 的输出策略是：

- A、B 都有效：写入四个真实坐标；
- 只有一边有效：另一边写 `-1 -1`；
- 两边都无效：整行写 `-1 -1 -1 -1`。

这样后续做外部筛选或几何求解时，不需要再额外维护索引映射，行号本身就对应固定的棋盘单元编号。

### 5. 导出带编号的匹配可视化

`export_visualization()` 会把两张图横向拼接，并对每个单元的导出点画标记：

- 绿色：两帧都成功匹配
- 黄色：单边存在
- 红色叉号：该单元缺失

这让算法结果不只停留在文本文件上，还能直接人工检查错配、漏配和边缘退化区域。

## 目录说明

```text
.
├── left/                     # 左相机样例输入
├── right/                    # 右相机样例输入
├── match_checker_cells.py    # 主脚本
├── requirements.txt          # Python 依赖
├── README.md                 # 项目说明
└── LICENSE                   # 开源许可证
```

## 说明

- 当前脚本中的输入文件路径是写死在常量 `LEFT_FILES` 和 `RIGHT_FILES` 里的。
- 如果要替换成你自己的图像数据，可以直接修改这两个常量，或者后续再把脚本改成命令行参数版本。
- 本仓库默认不跟踪生成产物，`match_vis_*.png` 和 `pnp_pointsAB_*.txt` 已加入 `.gitignore`。

## License

MIT
