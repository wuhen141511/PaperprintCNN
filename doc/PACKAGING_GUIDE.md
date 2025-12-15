# 部署包打包工具使用说明

## 📦 功能说明

`create_deployment_package.py` 是一个自动打包脚本，用于将服务器集成所需的所有文件打包成一个 ZIP 文件，方便发送给服务器对接同事。

## 🚀 使用方法

### 运行打包脚本

```bash
# 使用 uv
uv run python create_deployment_package.py

# 或使用 Python
python create_deployment_package.py
```

### 输出结果

脚本会生成一个带时间戳的 ZIP 文件：
```
qrcode_model_deployment_YYYYMMDD_HHMMSS.zip
```

例如：`qrcode_model_deployment_20251204_091013.zip`

## 📋 包含内容

打包文件包含以下内容：

### 必需文件（18个）

1. **核心预测器**
   - `predictor_singleton.py` - 预测器单例类

2. **模型文件**
   - `checkpoints/best_model.pth` - 模型权重（~102MB）
   - `checkpoints/class_names.json` - 类别配置

3. **源代码**
   - `src/__init__.py`
   - `src/inference.py` - 推理模块
   - `src/model.py` - 模型定义
   - `src/dataset.py` - 数据处理
   - `src/utils.py` - 工具函数

4. **文档**
   - `QUICKSTART.md` - 快速开始指南
   - `INTEGRATION_GUIDE.md` - 完整集成文档
   - `README.md` - 项目说明
   - `README_DEPLOYMENT.txt` - 部署说明（自动生成）

5. **示例代码**
   - `examples/flask_api.py` - Flask API 完整示例
   - `examples/fastapi_api.py` - FastAPI 完整示例
   - `examples/api_client.py` - 客户端调用示例

6. **测试工具**
   - `test_api.py` - API 测试脚本

7. **配置文件**
   - `config.yaml` - 配置示例（可选）
   - `requirements.txt` - Python 依赖（自动生成）

## 📊 打包信息

打包完成后会显示：

```
✅ 打包完成!
📦 包名称: qrcode_model_deployment_20251204_091013.zip
📊 文件数量: 18 个
💾 原始大小: 102.09 MB
🗜️  压缩后: 94.69 MB
📉 压缩率: 7.3%
```

## 🎯 使用场景

### 场景 1: 发送给服务器同事

```bash
# 1. 运行打包脚本
python create_deployment_package.py

# 2. 将生成的 ZIP 文件发送给同事
# qrcode_model_deployment_YYYYMMDD_HHMMSS.zip

# 3. 同事解压后即可使用
```

### 场景 2: 部署到新服务器

```bash
# 1. 打包
python create_deployment_package.py

# 2. 上传到服务器
scp qrcode_model_deployment_*.zip user@server:/path/to/deploy/

# 3. 在服务器上解压
unzip qrcode_model_deployment_*.zip

# 4. 按照 README_DEPLOYMENT.txt 进行集成
```

## 📝 脚本特性

### ✅ 自动检查

- 检查所有必需文件是否存在
- 显示每个文件的大小
- 如果缺少必需文件会报错并停止

### ✅ 自动生成

- `README_DEPLOYMENT.txt` - 详细的部署说明
- `requirements.txt` - Python 依赖列表（如果不存在）

### ✅ 智能压缩

- 使用 ZIP_DEFLATED 压缩算法
- 显示压缩前后大小对比
- 计算压缩率

### ✅ 详细信息

- 显示每个添加的文件
- 统计文件数量和大小
- 提供使用提示

## 🔧 自定义打包内容

如果需要修改打包内容，编辑 `create_deployment_package.py` 中的配置：

```python
# 必需文件列表
required_files = {
    'predictor_singleton.py': '核心预测器',
    'checkpoints/best_model.pth': '模型权重',
    # ... 添加或删除文件
}

# 可选文件
optional_files = {
    'requirements.txt': '依赖列表',
    # ... 添加或删除文件
}
```

## ❓ 常见问题

### Q: 打包失败，提示缺少文件？

**A:** 确保在项目根目录运行脚本，并且已经训练好模型（存在 `checkpoints/best_model.pth`）

### Q: 生成的 ZIP 文件太大？

**A:** 主要是模型文件（~102MB），这是正常的。如果需要减小大小，可以：
- 使用模型量化
- 使用更小的模型（如 MobileNet）

### Q: 如何验证打包是否成功？

**A:** 解压 ZIP 文件，检查是否包含所有必需文件，特别是：
- `predictor_singleton.py`
- `checkpoints/best_model.pth`
- `README_DEPLOYMENT.txt`

### Q: 可以自动上传到服务器吗？

**A:** 可以扩展脚本添加 SCP 或 FTP 上传功能，或使用：
```bash
scp qrcode_model_deployment_*.zip user@server:/path/
```

## 💡 最佳实践

1. **定期打包**: 每次模型更新后重新打包
2. **版本管理**: ZIP 文件名包含时间戳，便于版本追踪
3. **文档同步**: 确保文档与代码同步更新
4. **测试验证**: 打包后在干净环境中测试解压和集成

## 📞 技术支持

如有问题，请参考：
- `QUICKSTART.md` - 快速开始
- `INTEGRATION_GUIDE.md` - 完整指南
- `README_DEPLOYMENT.txt` - 部署说明（在 ZIP 包内）
