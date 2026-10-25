# Robot VLA server

GPU服务器工作区。代码由Git管理，数据、模型、日志和虚拟环境留在本机。

## 安装

```bash
git clone <你的仓库地址> robot-vla
cd robot-vla
./install.sh
./install.sh --check
```

当前目录即安装目录，无需target参数。安装器为robot-vla、同级的lerobot和openpi分别创建虚拟环境，并使用固定Git提交和依赖版本。已有外部仓库版本不匹配时会停止，避免覆盖本地修改。安装不会下载模型权重，也不会启动服务。已有数据、模型和配置不会被覆盖。

仅安装路由器环境（外部框架已部署时）：`./install.sh --skip-frameworks`。

## 配置

全部配置平铺在`config/`，格式均为JSON：

| 文件 | 用途 |
|---|---|
| server.json | 监听地址、端口、默认GPU、动作块长度、执行授权 |
| frameworks.json | 外部框架相对路径、仓库地址、固定提交、Python版本 |
| models.json | 模型注册、权重相对路径、模型可推理/执行状态 |
| action_codec.json | 客户端与服务端共享的六舵机动作协议 |
| dataset.json | 输入压缩包、校验和、数据转换及划分 |
| curation.json | 数据清洗与截断参数 |
| training.json | SmolVLA训练参数 |

配置中的相对路径以项目根目录为基准。GPU编号按容器可见设备编号解释，`--gpu`覆盖server.json中的默认值，在导入CUDA/JAX之前生效。

## 使用

```bash
./scripts/doctor.sh
./scripts/data.sh prepare
./scripts/data.sh check
./scripts/train.sh start --gpu 0
./scripts/train.sh status
./scripts/train.sh logs
./scripts/pi05.sh stats --gpu 0
./scripts/pi05.sh train --gpu 0 --steps 3000 --batch-size 4 --experiment red-block-v1
./scripts/service.sh start --gpu 0
./scripts/service.sh status
./scripts/service.sh logs
./scripts/service.sh stop
./scripts/service.sh start --gpu 0 --allow-motion
```

服务只监听127.0.0.1，客户端通过SSH隧道访问。默认关闭真实执行许可；配置`allow_motion`或显式`--allow-motion`可启用。训练和推理不会自动协调GPU资源，启动前由操作者分配空闲GPU。

模型权重分别放入`models/base/`及`models/trained/`，`models/active/`的软链接选择当前权重；模型名称及状态在`config/models.json`配置。模型缺失不阻碍软件安装，推理前必须准备相应权重。

## 目录

`src`核心代码，`config`配置，`scripts`操作入口，`tests`测试，`data/source`上传包，`data/datasets`训练数据集，`data/temp`中间数据，`models`模型，`logs`服务/训练/评估记录，`run`PID及worker socket。所有这些数据目录与`.venv`均被Git忽略。

升级：先停止服务，`git pull --ff-only`，运行`./install.sh`，检查后重新启动。Git会报告本地配置冲突，不会静默替换用户设置。仓库远端URL由仓库所有者设置，本地`git init`不会发布到GitHub。
