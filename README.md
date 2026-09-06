# DS-MSAN：ICBHI 呼吸音四分类

该项目将 ICBHI 呼吸周期分类为四类：`Normal`、`Crackle`、`Wheeze` 和
`Both`。当前主线使用统一的音频数据集、Mel 前端、DS-MSAN 模型和标准
ICBHI 指标。

## 主要修正

- official train 按患者划分内部 train/validation；
- official test 不参与训练、早停或权重选择；
- 训练和评估共用完全一致的 Mel 前端；
- 使用标准 ICBHI Score，同时仅为历史复现报告 Legacy Macro-OvR；
- checkpoint 保存配置、指标、epoch 和优化器状态；
- 现有 `0.6439` 裸 `state_dict` 仍可由 `evaluate.py` 严格加载。

## 文件结构

```text
dataset.py   ICBHI 音频读取、缓存、患者信息和数据视图
frontend.py  训练与评估共用的 MelSpectrogram/SpecAugment
metrics.py   标准 ICBHI Score 与历史 Macro-OvR 指标
model.py     DS-MSAN 网络
train.py     患者独立的训练/内部验证流程
evaluate.py  official test 独立评估、TTA 和混淆矩阵
tests/       指标、模型和旧权重兼容性测试
```

## 环境

需要 Python 3.10 或更高版本。`torch` 与 `torchaudio` 必须使用相同版本，
并根据服务器 CUDA 版本安装对应构建。

```bash
pip install -r requirements.txt
```

## metadata.csv

必须至少包含以下字段：

```text
filepath,onset,offset,crackles,wheezes,split
```

建议同时提供：

```text
patient,device
```

如果没有 `patient`，代码会从 ICBHI 文件名第一个下划线前提取患者编号；
未知设备统一记为 `-1`。

## 训练

```bash
python train.py \
  --audio-dir /path/to/audio_test_data \
  --csv-path /path/to/metadata.csv \
  --output-dir runs/dsmsan \
  --seed 24923
```

训练输出位于：

```text
runs/dsmsan/seed_24923/config.json
runs/dsmsan/seed_24923/best_checkpoint.pth
runs/dsmsan/seed_24923/last_checkpoint.pth
```

训练脚本只读取 `split=train`。内部验证采用患者独立划分，绝不会读取
`split=test`。

常用选项：

```bash
python train.py --help
python train.py --disable-amp
python train.py --no-cache
```

## Official test 评估

```bash
python evaluate.py \
  --checkpoint runs/dsmsan/seed_24923/best_checkpoint.pth \
  --audio-dir /path/to/audio_test_data \
  --csv-path /path/to/metadata.csv \
  --output-dir evaluation/seed_24923
```

评估会同时生成：

- 标准 ICBHI Score；
- Sensitivity、Specificity 和各类别召回率；
- 历史 Legacy Macro-OvR 分数；
- 分类报告、`metrics.json` 和归一化混淆矩阵。

默认不开启 TTA。需要单独观察 TTA 时：

```bash
python evaluate.py ... --tta --tta-shift 4
```

脚本会同时报告原始结果和 TTA 结果，不会用 TTA 覆盖原始结果。

## 评估旧 0.6439 权重

仓库中的旧权重是裸 `state_dict`，其文件名为：

```text
InnovativeResNet_best_DS_MSAN_seed_24923_score_0.6439.pth
```

可直接运行：

```bash
python evaluate.py \
  --checkpoint InnovativeResNet_best_DS_MSAN_seed_24923_score_0.6439.pth \
  --audio-dir /path/to/audio_test_data \
  --csv-path /path/to/metadata.csv
```

`0.6439` 来自旧版 Legacy Macro-OvR 指标，并且旧训练曾使用 test split 选权重，
因此不能当作修正后流程的独立 official test 成绩。旧权重仅用于历史复现。

## ICBHI Score

标准四分类指标为：

```text
SP = Normal 预测正确数 / Normal 总数
SE = 三种异常类别预测正确数之和 / 全部异常样本数
Score = (SE + SP) / 2
```

## 测试

```bash
pytest -q
```

模型结构、Mel 参数或音频预处理发生变化后必须重新训练；不能继续把旧权重
的文件名分数当作新实验结果。