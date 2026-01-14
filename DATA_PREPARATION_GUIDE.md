# PRIMER 数据准备指南

## 📋 目录
1. [数据格式要求](#数据格式要求)
2. [NC 转 NPY 转换](#nc-转-npy-转换)
3. [数据切片策略](#数据切片策略)
4. [完整工作流程](#完整工作流程)

---

## 1️⃣ 数据格式要求

### 训练数据必须是 `.npy` 格式！

虽然气象数据通常是 NetCDF (`.nc`) 格式，但 PRIMER 训练需要转换为 `.npy`：

**原因：**
- ✅ **10-100倍加载速度提升**：无需解析元数据
- ✅ **随机访问优化**：训练时需要随机读取大量文件
- ✅ **内存映射支持**：大数据集无需全部加载到内存
- ✅ **代码硬编码**：`load_data_util.py` 只接受 `.npy` 后缀

### 三种数据源的格式

| 数据源 | 形状 | 说明 |
|--------|------|------|
| **ERA5** | `(1, H, W)` | 单通道降水场 |
| **IMERG** | `(1, H, W)` | 单通道降水场 |
| **Gauge** | `(2, H, W)` | **双通道**：Channel 0=降水值, Channel 1=站点掩码 |

---

## 2️⃣ NC 转 NPY 转换

### 快速开始

```bash
# 1. 安装依赖
pip install xarray netcdf4 tqdm

# 2. 转换单个文件
python convert_nc_to_npy.py \
    --input /path/to/era5_2020.nc \
    --output ./data/ERA5_npy \
    --type era5 \
    --var tp

# 3. 批量转换目录
python convert_nc_to_npy.py \
    --input /path/to/nc_files/ \
    --output ./data/ERA5_npy \
    --type era5 \
    --var tp \
    --batch
```

### 完整参数说明

```bash
python convert_nc_to_npy.py \
    --input <nc文件或目录> \
    --output <输出目录> \
    --type <era5|imerg|gauge> \
    --var <变量名> \
    --lat-range <起始索引> <结束索引>  # 可选：空间裁剪
    --lon-range <起始索引> <结束索引>  # 可选：空间裁剪
    --batch  # 批量处理目录中所有文件
```

### 示例：处理不同数据源

#### ERA5 数据
```bash
# ERA5 降水数据（通常变量名为 'tp' 或 'precip'）
python convert_nc_to_npy.py \
    -i /data/ERA5/era5_2020_precip.nc \
    -o ./data/ERA5_npy \
    -t era5 \
    -v tp
```

#### IMERG 数据
```bash
# IMERG 卫星降水（通常变量名为 'precipitationCal'）
python convert_nc_to_npy.py \
    -i /data/IMERG/*.nc \
    -o ./data/IMERG_npy \
    -t imerg \
    -v precipitationCal \
    --batch
```

#### Gauge 观测数据
```bash
# Gauge 站点数据（需要降水值和站点掩码）
python convert_nc_to_npy.py \
    -i /data/gauges/gauge_2020.nc \
    -o ./data/gauges_npy \
    -t gauge \
    -v precip
```

---

## 3️⃣ 数据切片策略：按小时 vs 按月？

### ✅ **推荐：按小时切片（每个时刻一个文件）**

**原因：**

1. **训练效率最高**
   ```python
   # 代码每次随机选择一个文件
   file_path = random.choice(self.data_files)
   data = np.load(file_path)  # 只加载一个时刻
   ```
   - 如果是按月存储，需要读取整个月再切片 → **慢**
   - 按小时存储，直接加载目标时刻 → **快**

2. **内存友好**
   - 按月：可能需要加载几百个时刻（几GB）
   - 按小时：每次只加载 ~250KB

3. **支持任意长度训练**
   ```python
   # 配置文件中设置
   data.fixed_length = 10000  # 训练集大小
   ```
   - 可以从所有文件中随机采样
   - 无需担心月份不均衡

4. **代码设计就是为此优化的**
   ```python
   # load_data_util.py 会列举所有 .npy 文件
   self.data_files = [file for file in os.listdir(self.data_dir)
                      if file.endswith(".npy")]
   ```

### 文件命名规范

根据数据源采用不同命名格式：

| 数据源 | 文件名格式 | 示例 |
|--------|-----------|------|
| **ERA5** | `YYYYMMDDhh.npy` | `2020070413.npy` (2020年7月4日13时) |
| **IMERG** | `YYYYMMDD_HHMM.npy` | `20100821_2130.npy` (2010年8月21日21:30) |
| **Gauge** | `YYYYMMDDhh.npy` | `2017073118.npy` (2017年7月31日18时) |

**为什么不同格式？**
- ERA5/Gauge：通常按小时存档
- IMERG：半小时分辨率，需要分钟信息

### 如果数据量太大怎么办？

**场景：** 10年逐小时数据 = 87,600 个文件

**方案1：** 筛选有降水的时刻
```python
# 转换时跳过无降水的时刻
if data.max() < 0.1:  # 最大降水 < 0.1mm
    continue  # 不保存此文件
```

**方案2：** 使用数据过滤器（已内置）
```python
# training_config.py
def default_sample_filter_fn(x):
    """跳过无降水样本"""
    mask = x > 0.2
    if mask.sum() / mask.size < 0.02:  # 降水面积 < 2%
        return random.random() > 0.8  # 80%概率跳过
    return True
```

**方案3：** 按季节/月份训练
```bash
# 只转换夏季（主要降水季）
python convert_nc_to_npy.py \
    -i /data/ERA5/era5_2020_06-08.nc \
    -o ./data/ERA5_npy_summer \
    -t era5
```

---

## 4️⃣ 完整工作流程

### Step 1: 准备原始 NetCDF 数据

确保你的 NetCDF 文件包含：
- 时间维度：`time`
- 空间维度：`lat`, `lon` (或 `latitude`, `longitude`)
- 降水变量：`tp`, `precip`, `precipitation` 等

```python
# 检查 NetCDF 结构
import xarray as xr
ds = xr.open_dataset("your_data.nc")
print(ds)  # 查看变量名和维度
```

### Step 2: 转换为 NPY

```bash
# 批量转换（推荐）
python convert_nc_to_npy.py \
    --input /path/to/your/nc_files/ \
    --output ./data/your_data_npy \
    --type era5 \  # 或 imerg, gauge
    --var <你的降水变量名> \
    --batch
```

### Step 3: 计算归一化参数

```python
# 创建脚本 compute_normalization.py
import numpy as np
from pathlib import Path
from tqdm import tqdm

data_dir = Path("./data/your_data_npy")
npy_files = list(data_dir.glob("*.npy"))

# 增量计算统计量（避免内存溢出）
sum_val = 0
sum_sq = 0
count = 0
min_val = float('inf')
max_val = float('-inf')

for file in tqdm(npy_files):
    data = np.load(file)

    # 只统计第一个通道（降水值）
    if data.ndim == 3:
        data = data[0]

    # 去除 NaN
    valid_data = data[~np.isnan(data)]

    sum_val += valid_data.sum()
    sum_sq += (valid_data ** 2).sum()
    count += valid_data.size
    min_val = min(min_val, valid_data.min())
    max_val = max(max_val, valid_data.max())

mean = sum_val / count
std = np.sqrt(sum_sq / count - mean ** 2)

# 保存结果
results = {
    'min_val': float(min_val),
    'max_val': float(max_val),
    'mean': float(mean),
    'std': float(std),
    'clip_min': -3.0,  # 标准归一化裁剪范围
    'clip_max': 3.0,
}

np.save('property_results.npy', results)
print(f"归一化参数已保存:")
print(f"  min: {min_val:.4f}")
print(f"  max: {max_val:.4f}")
print(f"  mean: {mean:.4f}")
print(f"  std: {std:.4f}")
```

### Step 4: 更新配置文件

```python
# code/configs/training_config.py

# 数据路径
data.root_dir = "./data/your_data_npy"

# 加载归一化参数
import numpy as np
norm_params = np.load('./property_results.npy', allow_pickle=True).item()

data.min_val = norm_params['min_val']
data.max_val = norm_params['max_val']
data.mean = norm_params['mean']
data.std = norm_params['std']
data.clip_min = norm_params['clip_min']
data.clip_max = norm_params['clip_max']
data.normalization = 'standard'  # 推荐使用 standard
```

### Step 5: 开始训练

```bash
# 单源训练
python code/train_with_multiple_gpu.py

# 多源训练
python code/train_with_multiple_gpu_mutiple_sources.py
```

---

## 📊 数据量估算

### 存储空间计算

假设数据分辨率 250×250：

```
单个时刻大小 = 250 × 250 × 4 bytes (float32) = 250 KB
```

| 数据量 | 文件数 | 存储空间 |
|--------|--------|---------|
| 1个月（逐小时） | 720 | ~180 MB |
| 1年（逐小时） | 8,760 | ~2.2 GB |
| 10年（逐小时） | 87,600 | ~22 GB |
| 10年（每3小时） | 29,200 | ~7.3 GB |

### 训练集大小建议

```python
# training_config.py
data.fixed_length = 10000  # 推荐值

# 这意味着：
# - 每个 epoch 训练 10000 个样本
# - 从所有 .npy 文件中随机采样
# - 可以小于实际文件数（重复采样）
# - 可以大于实际文件数（允许重复）
```

---

## ⚠️ 常见问题

### Q1: 我的数据是分钟级别的，怎么办？
**A:** 可以保留分钟信息在文件名中：
```
YYYYMMDD_HHMM.npy  # 如 20200704_1330.npy
```
代码只关心文件后缀是 `.npy`，不解析文件名。

### Q2: 不同年份的数据放一起还是分开？
**A:** 放一起！代码会自动读取目录下所有 `.npy` 文件：
```
data/ERA5_npy/
├── 2018010100.npy
├── 2018010101.npy
├── ...
├── 2020123123.npy
```

### Q3: 需要排序文件名吗？
**A:** 不需要！训练时是随机选择文件：
```python
file_path = random.choice(self.data_files)
```
时间顺序无关紧要。

### Q4: Gauge 数据的掩码怎么创建？
**A:** 三种方法：

**方法1：** 如果有站点坐标列表
```python
mask = np.zeros((H, W))
for station in stations:
    i, j = lonlat_to_grid(station.lon, station.lat)
    mask[i, j] = 1  # 或站点密度
```

**方法2：** 从降水值反推（如果已插值）
```python
# 假设插值前未观测位置是 NaN
mask = (~np.isnan(precip_data)).astype(np.float32)
```

**方法3：** 使用站点影响半径
```python
for station in stations:
    i, j = lonlat_to_grid(station.lon, station.lat)
    mask[i-r:i+r, j-r:j+r] = 1  # r = 影响半径
```

### Q5: 空间分辨率不是 250×250 可以吗？
**A:** 可以！修改配置：
```python
data.expected_img_size = (1, 512, 512)  # 任意分辨率
```
但需要：
- 所有文件相同分辨率
- 更大分辨率 → 更多显存/计算时间

---

## 📝 检查清单

准备数据前请确认：

- [ ] 所有数据已转换为 `.npy` 格式
- [ ] ERA5/IMERG 形状为 `(1, H, W)`
- [ ] Gauge 形状为 `(2, H, W)`，第二通道是掩码
- [ ] 文件命名符合规范（包含时间戳）
- [ ] 所有文件空间分辨率一致
- [ ] 已计算并保存归一化参数
- [ ] 配置文件已更新数据路径和归一化参数
- [ ] 目录结构正确：
  ```
  PRIMER/
  ├── data/
  │   └── your_data_npy/
  │       ├── 2020010100.npy
  │       ├── 2020010101.npy
  │       └── ...
  ├── code/
  │   └── configs/
  │       └── training_config.py  # 已更新
  └── property_results.npy  # 归一化参数
  ```

---

## 🚀 快速参考

```bash
# 1. 转换数据
python convert_nc_to_npy.py -i /data/nc/ -o ./data/npy/ -t era5 -v tp --batch

# 2. 计算归一化
python compute_normalization.py

# 3. 更新配置
# 编辑 code/configs/training_config.py

# 4. 开始训练
python code/train_with_multiple_gpu.py
```

---

**有问题？** 查看示例数据：
- `data/ERA5_npy/2020070413.npy` - ERA5 示例
- `data/IMERG_npy/20100821_2130.npy` - IMERG 示例
- `data/gauges_npy/2017073118.npy` - Gauge 示例（双通道）
- `code/process_gauges_before/explore_gauges_dataset.ipynb` - 预处理示例
