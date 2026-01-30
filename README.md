# MCM-A-2026

本项目实现 2026 MCM Problem A：智能手机电池放电的高保真热-电-退化耦合模型，包含场景仿真与论文级可视化。

## 功能概览
- SOC / 温度 / SOH 耦合 ODE 动力学
- 电学：OCV 曲线 + 内阻 Arrhenius + 二次方程求电流
- 热学：I^2R + 负载热（扣除辐射逃逸）+ 对流散热
- 退化：循环老化 + 日历老化（Arrhenius）
- 断电截止（V_term < V_cutoff）事件终止
- 四类图：相图、压降分析、灵敏度热力图、功率分解
- 通信与导航采用宏观状态机（WiFi/GPS）

## 目录结构
- `solution.py`：主程序（仿真 + 可视化）
- `config.yaml`：全部参数与场景配置
- `requirements.txt`：依赖列表

## 环境与依赖
建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 快速运行
```bash
python solution.py --config config.yaml
```

## Data Sources & Parameter Estimation
为满足“数据驱动”要求，项目引入权威公开数据集，并提供参数拟合与验证脚本。

**数据集来源**
- CALCE Battery Data (UMD): https://web.calce.umd.edu/batteries/data/
- NASA PCoE Randomized Battery Usage (Zenodo): https://zenodo.org/records/15277374
- Panasonic 18650PF (Mendeley Data): https://data.mendeley.com/datasets/wykht8y7tg/1
（本地文件与下载链接索引见 `datasets/SOURCES.md`）

**OCV 参数拟合**
```bash
python fit_params.py --ocv_csv datasets/calce_ocv/cs2_8_ocv_curve.csv
```
拟合结果会输出 `ocv_coeffs`，填入 `config.yaml` 的 `battery_params.ocv_coeffs`。

**温度参数拟合（Ea_R / Ea_cap）**
```bash
python derive_params.py
```
默认从 Panasonic 18650PF 多温度循环数据拟合 `Ea_R` 与 `Ea_cap`，并输出拟合图到 `outputs/`。

**功耗参数提取（AOSP power_profile.xml）**
```bash
python extract_power_profile.py --voltage 3.85
```
注意：AOSP 基准 profile 的 mA 值是占位符，仅用于流程演示；应替换为目标设备测量值。

**随机负载验证**
```bash
python solution.py --config config.yaml
```
当 `validation.enabled: true` 且配置了 `validation.current_profile_csv`，
程序会生成“测量电压 vs 模型电压”对比图用于验证。

## 结果输出（CSV/JSON）
运行后会自动把每个场景的详细时间序列输出到 `output.directory`（默认 `outputs/`）。  
可在 `config.yaml` 中设置：
```yaml
output:
  enabled: true
  format: csv   # csv or json
  directory: outputs
```

## 实时可视化（SCI 论文美学）
支持运行时逐步绘制多面板仪表盘（全英文标签与期刊风格）。
```yaml
visualization:
  realtime: true
  interval_ms: 80
  max_points: 600
```

## 配置说明（YAML）
所有参数均在 `config.yaml` 中集中管理，包含：
- `battery_params`：电、热、退化、组件、初始状态参数
- `scenarios`：多个使用场景（Video/Gaming/Winter）
- `sensitivity`：亮度与环境温度的网格敏感性分析

### Profile 配置
场景中的 `brightness/apl/cpu_util/wifi_state/gps_state/charger_power` 支持两类配置：

1. 常量：
```yaml
brightness: {type: constant, value: 0.85}
```

2. 分段常量（piecewise）：
```yaml
cpu_util:
  type: piecewise
  segments:
    - {start: 0, end: 1200, value: 0.2}
    - {start: 1200, end: 3600, value: 0.8}
    - {start: 3600, end: null, value: 0.4}
```

## 输出图像
运行后会弹出并显示四类论文级图像：
- Death Spiral 相图（SOC vs 温度）
- 压降与截止分析图（V_term 与 V_ocv）
- 灵敏度热力图（亮度×环境温度 -> 续航）
- 功率分解堆叠面积图

## 常见问题
- **没有 SciencePlots 样式？** 如果系统未安装 `science` 风格，将自动回退到 seaborn 的论文风格。
- **想保存图片？** 可以在 `solution.py` 中替换 `plt.show()` 为 `plt.savefig(...)` 或添加 `plt.savefig`。

## 备注
本模型用于学术建模与仿真研究，参数为合理默认值，可根据实测数据在 `config.yaml` 中调整。
