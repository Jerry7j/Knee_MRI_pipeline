# knee_mri 包

[English](README.md) | [简体中文](README.zh-CN.md)

四个业务模块通过统一命令行组织 MRI 分割流程：

| 模块 | 职责 |
| --- | --- |
| `data` | 输入读取、数据集、患者划分、YOLO 定位和配对裁剪 |
| `models` | 2.5D/3D U-Net 构建和权重加载 |
| `engine` | 训练、验证、测试、推理、指标和概率融合 |
| `visualization` | 分割对比、叠加、熵图和训练曲线 |

`common.py` 集中提供 JSON 读写、配置校验、模型文件检查、随机种子和设备选择。相对路径按配置的 project_root 解析；JSON 写入自动创建父目录；auto 设备优先使用 cuda:0，否则使用 CPU。

`cli.py` 提供 12 个命令。`python -m knee_mri --help` 与安装后的 `knee-mri --help` 等价。check、split 和命令帮助仅依赖标准库。导入包不会启动训练、加载数据或下载模型。

函数和方法使用英文文档字符串说明功能；业务接口标注参数、返回值、数组形状与主要异常，简单协议方法和测试方法使用简要说明。
