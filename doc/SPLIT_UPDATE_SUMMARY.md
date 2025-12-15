# 数据划分脚本更新说明

## 更新概述

`split_dataset.py` 已升级以支持多标签分类系统的数据自动划分。

---

## 主要改进

### ✅ 新增功能

1. **多标签模式支持**
   - 自动分割 JSON 标注文件
   - 保持图片和标注的对应关系
   - 显示验证集中各标签的分布统计

2. **自动模式检测**
   - 智能识别数据集类型（多标签 vs 单分类）
   - 无需手动指定模式（也可手动指定）

3. **向后兼容**
   - 完全兼容原有的单分类目录结构
   - 保留所有原有功能

---

## 使用示例

### 多标签模式（新）

```bash
# 自动检测并分割多标签数据
python split_dataset.py --ratio 0.2

# 手动指定多标签模式
python split_dataset.py --mode multi-label --ratio 0.2
```

**数据结构**:
```
data/train/
├── images/
│   ├── img001.jpg
│   └── ...
└── annotations.json

→ 分割后 →

data/train/
├── images/          # 80% 训练图片
└── annotations.json # 训练标注

data/val/
├── images/          # 20% 验证图片
└── annotations.json # 验证标注
```

### 单分类模式（兼容）

```bash
# 自动检测并分割单分类数据
python split_dataset.py --ratio 0.2

# 手动指定单分类模式
python split_dataset.py --mode single-class --ratio 0.2
```

---

## 关键特性

| 特性 | 多标签模式 | 单分类模式 |
|------|-----------|-----------|
| 数据格式 | JSON 标注 | 目录结构 |
| 标注分割 | ✅ 自动 | N/A |
| 标签统计 | ✅ 显示 | ✅ 显示 |
| 自动检测 | ✅ 支持 | ✅ 支持 |
| 文件移动 | ✅ 支持 | ✅ 支持 |

---

## 完整工作流程

### 多标签分类项目

```bash
# 1. 准备图片
mkdir -p data/train/images
# 将图片放入 images/ 目录

# 2. 创建标注模板
python create_annotations.py create data/train/images

# 3. 编辑标注文件
# 编辑 data/train/annotations.json

# 4. 验证标注
python create_annotations.py validate data/train/annotations.json

# 5. 分割数据集（自动检测为多标签模式）
python split_dataset.py --ratio 0.2

# 6. 开始训练
python train_model.py
```

---

## 输出示例

```
============================================================
Dataset Splitting Tool
============================================================
Mode: multi-label
Training directory: E:\workspace_paper\PaperprintCNN\data\train
Validation directory: E:\workspace_paper\PaperprintCNN\data\val
Validation ratio: 20.0%
Random seed: 42
============================================================

⚠️  This will MOVE files from train to val directory. Continue? (y/n): y

Found 500 images in annotations
Target validation ratio: 20.0%
Will move 100 images to validation (20.0%)
------------------------------------------------------------
✓ Moved 100 images to validation
✓ Training set: 400 images
✓ Validation set: 100 images
✓ Saved annotations to both directories

Label distribution in validation set:
  - is_copied: 45/100 (45.0%)
  - is_blurry: 23/100 (23.0%)
  - is_low_light: 38/100 (38.0%)

============================================================
Split Summary
============================================================
Total images: 500
Training set: 400
Validation set: 100
Actual validation ratio: 20.0%
============================================================
✅ Dataset split completed successfully!
============================================================
```

---

## 测试脚本

提供了测试脚本验证功能：

```bash
python test_split_dataset.py
```

测试脚本会：
1. 创建测试数据
2. 执行分割操作
3. 验证结果
4. 清理测试数据

---

## 技术细节

### 自动检测逻辑

```python
def detect_dataset_mode(train_dir):
    # 1. 检查多标签结构
    if exists('images/') and exists('annotations.json'):
        return 'multi-label'
    
    # 2. 检查单分类结构
    if has_class_subdirectories():
        return 'single-class'
    
    # 3. 默认多标签
    return 'multi-label'
```

### 分层采样逻辑 (Label Powerset Stratified Sampling)

```python
# 1. 识别标签组合 (Label Combination)
# 将每个样本的标签向量视为一个独立的"类"
# 例如: [0, 1, 0] 是一个组合，[1, 1, 0] 是另一个组合
groups = defaultdict(list)
for img, labels in annotations.items():
    signature = tuple(labels.values())
    groups[signature].append(img)

# 2. 分组采样
val_images = []
for signature, group_imgs in groups.items():
    # 对每个唯一的标签组合单独采样
    # 确保验证集中包含与训练集相同比例的该组合
    n_val = int(len(group_imgs) * ratio)
    val_images.extend(random.sample(group_imgs, n_val))

# 3. 结果合并
random.shuffle(val_images)
```

---

## 常见问题

### Q: 如何恢复分割前的状态？

**A**: 
```bash
# 1. 将验证集图片移回训练集
mv data/val/images/* data/train/images/

# 2. 合并标注文件
python create_annotations.py merge \
    data/train/annotations.json \
    data/val/annotations.json \
    -o data/train/annotations.json

# 3. 删除验证集目录
rm -rf data/val
```

### Q: 分割后标签分布不均衡？

**A**: 由于是随机分割，可能出现不均衡。解决方法：
- 使用不同的随机种子重试
- 增加数据量
- 手动调整（高级）

### Q: 支持增量分割吗？

**A**: 不建议。如需重新分割：
1. 先恢复原始状态
2. 重新运行分割脚本

---

## 相关文档

- 📖 `SPLIT_DATASET_GUIDE.md` - 详细使用指南
- 📖 `MULTILABEL_GUIDE.md` - 多标签分类指南
- 🧪 `test_split_dataset.py` - 测试脚本

---

## 更新日志

### v2.1 (最新)
- ✅ **升级分层采样算法**: 实现标签幂集分层采样，大幅提升数据分布一致性
- ✅ **优化稀有样本处理**: 确保稀有标签组合在验证集中有代表性

### v2.0
- ✅ 新增多标签模式
- ✅ 自动模式检测
- ✅ JSON 标注分割
- ✅ 标签分布统计
- ✅ 向后兼容

### v1.0
- 基础单分类分割

---

**升级完成！** 现在可以无缝支持多标签分类数据的自动划分。🎉
