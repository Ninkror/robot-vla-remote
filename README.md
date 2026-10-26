# Robot VLA 服务端使用说明

本项目在GPU服务器上整理训练数据、训练和评估 SmolVLA/π0.5，并为机器人工作台提供远程推理。服务器返回动作链，真实机械臂动作由客户端发布。

当前模型输入为单RGB图像、任务和六维真实舵机状态，输出为六维绝对脉冲动作链，舵机顺序 `[1,2,3,4,5,10]`。默认动作块为72行，每行对应150 ms。

架构、数据契约和测试方法见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 1. 已部署服务器的常用操作

```bash
ssh robot-vla
cd /root/robot-vla
./scripts/service.sh status
./scripts/service.sh start --gpu 0
./scripts/service.sh logs --lines 80
./scripts/service.sh stop
```

以上示例的 `start`在默认 `config/server.json`中仅开放推理。需要允许客户端执行模型动作时，先停止已运行服务，再启动：

```bash
./scripts/service.sh start --gpu 0 --allow-motion
```

`start`不会替换已经运行的服务。如果要更换GPU或执行授权，先 `stop`再 `start`。要恢复为仅推理，确保 `config/server.json`的 `allow_motion`为 `false`，再不带 `--allow-motion`启动；配置为true时省略参数并不能关闭许可。

`SERVICE_STARTED`只代表路由器进程启动，模型还可能加载中。客户端“推理服务端”就绪或成功返回推理，才说明请求链路可用。`status`在停止状态返回退出码1，这本身不是启动失败。

查看GPU占用：

```bash
nvidia-smi
```

GPU编号按当前容器可见编号解释。`--gpu`覆盖 `config/server.json`中的默认 `gpu`，安装时不绑定GPU。项目不自动协调实验室其他用户的显卡任务，也不会因为 `service.sh stop`而停止其他项目或后台训练。

## 2. 在另一台服务器安装

需要 Linux、Python 3.10～3.12、Git、可用的venv/pip、网络及匹配PyTorch/JAX CUDA依赖的NVIDIA驱动。容器必须能看到GPU。当前实验室部署已在容器中完成，不要求用户重新创建Docker。

```bash
git clone https://github.com/Ninkror/robot-vla-remote.git robot-vla
cd robot-vla
./install.sh
./install.sh --check
```

也可从自己托管的Git地址克隆。`install.sh`以当前仓库位置为安装根目录，不需要 `--target`。

安装器创建三套独立虚拟环境：

```text
robot-vla/.venv/    路由、任务入口、安装与服务管理
lerobot/.venv/      SmolVLA、数据转换、离线评估
openpi/.venv/       π0.5、JAX及OpenPI相关任务
```

LeRobot、OpenPI默认与本仓库平级，路径、Git提交和Python版本写在 `config/frameworks.json`。已有仓库的提交不匹配时安装会停止，不自动切换已有目录的版本。安装需要访问依赖源，且不自动下载模型权重、不复制数据、不启动推理服务。

如果外部框架已经准备好，只安装本项目环境：

```bash
./install.sh --skip-frameworks
```

`--check`检查配置、解释器是否存在和框架Git版本；进一步运行：

```bash
./scripts/doctor.sh
```

`doctor.sh`会导入PyTorch/JAX并探测GPU、模型资产和数据包校验和，但不启动推理服务或训练。它的“压缩包校验通过”只表示文件哈希匹配，不表示数据schema满足当前转换流程。

## 3. 配置与目录

所有配置为平铺的 `config/*.json`，项目内部相对路径以仓库根目录解析。

| 文件 | 用途 |
|---|---|
| `server.json` | 本机监听地址/端口、默认GPU、72行动作块、执行许可 |
| `frameworks.json` | LeRobot/OpenPI路径、固定Git提交、Python版本 |
| `models.json` | 模型ID、框架、checkpoint、默认模型、推理/执行状态 |
| `action_codec.json` | 两端共享的动作编码、舵机顺序和限位 |
| `dataset.json` | 输入压缩包、SHA256、数据集名、转换及划分 |
| `training.json` | SmolVLA参数及 `pi05`子对象中的π0.5默认参数 |
| `curation.json` | 可选的人工轨迹截断与数据合并参数 |

```text
data/source/       客户端上传的原始tar.zst包
data/temp/         转换解包及清洗的中间数据
data/datasets/     整理好的LeRobot数据集
models/base/       基础模型、VLM及缓存资产
models/trained/    训练checkpoint
models/active/     激活模型的软链接
models/assets/     π0.5归一化统计等资产
logs/             服务日志、训练状态、离线评估/推理结果
run/              PID、worker socket和状态文件
```

`data/`、`models/`、`logs/`、`run/`、`.venv/`被Git忽略，克隆不会携带。基础权重准备位置为 `models/base/smolvla/`、`models/base/smolvlm2/`和 `models/base/pi05/`，相关缓存位于 `models/base/huggingface/`、`models/base/openpi/`。推理和训练入口默认使用离线模型加载，准备好本地资产后再运行。

## 4. 接收并整理新数据

客户端上传包默认进入 `/root/robot-vla/data/source/`。当前转换仅接受：上传包顶层schema v2，内部episode schema v7，输入配置 `rgb_state_v1`。旧v6包即使校验和正确也不能直接转换。

1. 在客户端保存并上传成功episode。
2. 服务器检查上传包的校验和：

```bash
cd /root/robot-vla
sha256sum data/source/export-实际包名.tar.zst
```

3. 编辑 `config/dataset.json`，填写 `source_archive`、`source_sha256`和新的 `name`，检查 `validation_episode_ids`等划分设置。新的数据批次使用新名字，避免覆盖已完成的数据集。文件中保留的旧批次配置不是新训练的自动输入。
4. 执行：

```bash
./scripts/data.sh prepare
./scripts/data.sh check
```

输出为 `data/datasets/<name>/`，其中 `quality-report.json`记录划分、帧数和动作投影等信息。先检查报告再训练，尤其要确认训练和验证episode均存在、没有重复泄漏。

转换保留RGB和六维状态，按照150 ms重采样，再编码训练动作。当前允许的相邻图像间隔为1200 ms，最近图像重采样容差为600 ms；不会把一个episode自动切成多个任务片段。默认专家末端可按 `terminal_hold_rows`补充确认夹持后的终态行。

转换失败留下中间目录时：

```bash
./scripts/data.sh clean
```

这个命令只清理当前配置名对应的 `data/temp/`，不删除原始上传包和已生成的数据集。它不会使已有 `data/datasets/<name>`可被覆盖。

## 5. 训练SmolVLA

准备基础权重和通过检查的数据集，编辑 `config/training.json`后：

```bash
./scripts/train.sh start --gpu 0
./scripts/train.sh status
./scripts/train.sh logs --lines 80
```

该命令后台启动训练。输出权重位于 `models/trained/<run_name>/checkpoints/.../pretrained_model/`，训练状态、命令和日志位于 `logs/training/<run_name>/`。

启动成功不等于训练完成，按status中的 `completed`和checkpoint检查结果。当前入口没有训练 `stop`子命令；`service.sh stop`只管理推理服务，不管理训练。

## 6. 训练和评估π0.5

π0.5使用同一份 `dataset.json`选择数据集，`training.json`的 `pi05`子对象提供默认steps、batch_size和experiment。先计算训练划分的归一化统计，再训练：

```bash
./scripts/pi05.sh stats --gpu 0
./scripts/pi05.sh train \
  --gpu 0 \
  --steps 3000 \
  --batch-size 4 \
  --experiment red-block-v1
```

这两个命令在前台运行，断开会话前应使用自己已有的会话管理方式。checkpoint位于 `models/trained/pi05_rosorin_red_block/<experiment>/<step>/`，不要把 `params/`子目录误设为激活路径。要重复训练，优先使用新的experiment名；`--overwrite`会允许覆盖同名实验，不作为默认操作。

对已有π0.5 checkpoint执行离线评估：

```bash
./scripts/pi05.sh evaluate \
  --gpu 0 \
  --checkpoint models/trained/pi05_rosorin_red_block/red-block-v1/2999 \
  --output logs/pi05/evaluation.json
```

该评估使用数据集验证episode的锚点/动作窗口，并不证明真实抓取成功率。

## 7. 离线推理与模型激活

对SmolVLA做一次离线推理：

```bash
./scripts/infer.sh \
  --gpu 0 \
  --model models/active/smolvla \
  --image /path/to/rgb.jpg \
  --state '[500,725,49,149,500,498]' \
  --task '抓取红色方块'
```

结果写入 `logs/inference/`。以上状态是示例输入，不是读取当前机械臂。

对一条v7专家记录比较预测与专家目标：

```bash
./scripts/evaluate.sh \
  --gpu 0 \
  --model models/active/smolvla \
  --episode /path/to/episode_实际编号
```

激活模型的操作是在服务停止后，让 `models/active/smolvla`或 `models/active/pi05`软链接指向相应的完整checkpoint目录，并检查 `config/models.json`。例如，确认目标checkpoint存在且当前active路径是软链接或尚不存在后：

```bash
ln -sfn /root/robot-vla/models/trained/实际SmolVLA运行/checkpoints/实际步数/pretrained_model models/active/smolvla
```

模型加载路径改变后需要重新启动服务。模型列表中的 `trained_dry_run`用于仅推理；`validated`还需 `robot_compatible=true`及服务级许可才可能允许执行。不能仅通过改状态标签把未经适配的基础权重变成可执行模型。

## 8. 连接客户端

服务只监听 `127.0.0.1:8081`，客户端经SSH隧道访问，不需要为gRPC另外开放Docker端口。

客户端执行：

```bash
./scripts/configure_server.sh --host 10.130.151.21 --ssh-port 8167 --user root
./scripts/configure_server.sh --check
```

客户端选择 `model_id`，路由器按需切换模型worker，同一时刻只保持一个活跃模型。首次加载可能较慢；不要把加载中当成网络断开，也不要在另一个执行会话持有模型时强制切换。

## 9. 常见问题与升级

| 问题 | 检查项 |
|---|---|
| SSH成功但推理不可用 | 服务是否启动、8081是否一致、worker是否ready |
| `router has no active model`或加载失败 | `logs/service.log`和 `logs/worker-<model_id>.log`，checkpoint与框架环境 |
| codec不一致 | 客户端、服务配置及模型资产中的codec是否同源、字节哈希是否一致 |
| `schema_version 7`错误 | 输入的是旧数据或不完整episode，不能靠改版本号修复 |
| 数据集已存在 | 使用新数据集名；`data.sh clean`不删除最终数据集 |
| CUDA显存不足 | 查看全部GPU占用，运行时选择空闲GPU；安装不预留显存 |
| 动作不能执行 | 检查模型状态、robot_compatible、服务许可，修改后重新推理 |

本仓库可以直接经SSH克隆；是否另设Git托管远端由仓库所有者决定。配置了上游后，升级通常为停止服务、提交或备份本机配置、`git pull --ff-only`、`./install.sh`、自检、重新启动。Git报告冲突时先处理冲突，不覆盖数据和模型目录。
