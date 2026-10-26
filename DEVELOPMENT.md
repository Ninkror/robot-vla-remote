# Robot VLA 服务端开发说明

日常安装、数据转换、训练及服务命令见 [README.md](README.md)。本项目独立于机器人客户端和Agentic OS，负责将LeRobot/OpenPI能力适配到当前六舵机机器人契约。

## 1. 代码与运行边界

```text
robot-vla/      本项目Git仓库、路由器、数据与训练入口
lerobot/        固定版本上游代码及独立.venv
openpi/         固定版本上游代码及独立.venv
```

| 目录 | 职责 |
|---|---|
| `src/common/` | 根路径、JSON配置、环境安装、自检、运行时GPU与解释器选择 |
| `src/action/` | 动态有界相对动作编码/解码 |
| `src/dataset/` | 严格v7读取、重采样、质量报告、终态扩展、LeRobot格式及可选清洗 |
| `src/training/` | SmolVLA训练启动、后台worker、状态与checkpoint契约 |
| `src/pi05/` | OpenPI训练配置、v3数据加载、变换、统计、训练和评估 |
| `src/service/` | 服务控制、gRPC入口、输入输出校验、SmolVLA runner |
| `src/router/` | 模型注册表、单活跃worker、会话锁和模型状态 |
| `src/workers/` | 跨虚拟环境worker进程和Unix socket通信 |
| `src/proto/` | 与客户端一致的协议定义及生成绑定 |
| `src/inference/`、`src/evaluation/` | 离线推理和专家目标评估 |
| `config/` | 全部JSON配置 |
| `scripts/` | 薄入口脚本 |
| `tests/unit/` | 编码、数据、协议约束和部署行为测试 |

`common.paths.ROOT`从源文件位置推导。`CONFIGS`对应 `config/`，`RUNTIME`对应 `run/`，`RUNS`当前是 `logs/`的内部别名；不要据此重新创建旧的 `configs/`、`runtime/`或 `runs/`目录。

## 2. 三套Python环境

| 环境 | 典型入口 | 原因 |
|---|---|---|
| `robot-vla/.venv` | service控制、router、安装、训练启动器 | 保持路由层依赖轻量 |
| `lerobot/.venv` | 数据转换、SmolVLA训练/推理、目标评估 | PyTorch/LeRobot依赖 |
| `openpi/.venv` | π0.5统计、训练、评估、模型worker | JAX/OpenPI依赖 |

`scripts/*.sh`先进入本项目的 `common.launch`。它消费 `--gpu`、设置 `CUDA_VISIBLE_DEVICES`和模型缓存环境，再exec到正确解释器。必须在导入CUDA/JAX之前完成GPU选择。服务和SmolVLA后台训练还要将所选GPU传递给子进程，不能只在启动shell中修改一个局部变量。

GPU默认来自 `server.json`；`models.json`不按模型重复保存GPU。当前策略为 `single_active`，传入多个设备编号不代表已经实现多卡训练或多worker并行调度。

安装器位于 `common/install.py`，`install.sh`仅负责定位根目录。固定依赖来自三份lock文件，外部仓库提交来自 `frameworks.json`。修改框架版本时要同时验证API、数据格式、权重加载与锁定依赖，而不是只改一个Git哈希。

## 3. 数据链路

```text
客户端v7 episode
→ 顶层schema v2的tar.zst上传包
→ data/source
→ 校验包哈希、成员路径及episode文件哈希
→ data/temp/<name>/archive和session
→ 独立状态/动作流配合RGB重采样
→ 有界相对动作编码
→ data/datasets/<name>和quality-report.json
```

`dataset/raw.py`只接受v7，不提供旧版本兼容分支。绝对脉冲原始状态/动作顺序为 `[1,2,3,4,5,10]`。RGB、状态、动作及时间来源不可缺失；专家记录还需成功确认、锚点和专家目标证据。修改schema时同步客户端writer、上传校验和服务器reader，不能只修改manifest中的版本数字。

原始RGB时间作为图像锚点，客户端已保存最近状态及最新历史动作；服务器在150 ms训练网格上重新选择RGB，同时独立选择状态/动作，不能把某张图像上的旧配对直接当成网格时刻的真实事件。当前重采样RGB容差为600 ms，状态偏差上限100 ms，未来动作查找上限225 ms；这些是不同约束。

`RESAMPLE_TOLERANCE_NS`目前是 `dataset/converter.py`内常量，不是dataset.json的配置字段。`MAX_STATE_SKEW_NS`和动作lookahead位于 `dataset/raw.py`。修改阈值要结合客户端及测试评估，不靠放大单个值掩盖采集错误。

LeRobot特征包括 `observation.images.arm_camera`、`observation.state`、`action`和任务。深度不进入这些训练特征；专家目标坐标可用于离线目标误差分析。`rename_map`负责SmolVLA训练时的相机特征名适配，推理侧统一成 `camera1`。

`quality-report.json`记录训练/验证索引、样本投影、source重复及终态补齐。`terminal_hold_rows`只适用于已确认闭合的专家记录：扩展到最后有效事件，并追加指定数量的保持行，不凭空把失败episode标成成功。训练source重复不能复制验证episode进训练集。

## 4. 动作表示和模型资产

原始记录及网络返回都是六维绝对舵机脉冲。模型训练目标和模型内部输出是 `dynamic_bounded_relative_v1`编码，runner按当前状态解码为绝对动作。

`action/codec.py`按软硬限位及每步max_step构造动态区间；客户端使用对应codec验证解码后的动作。不能把训练潜变量直接发给机器人，也不能在客户端对绝对动作再次累加。

当前 `action_codec.json`主要字段：

| 字段 | 含义 |
|---|---|
| `schema`、`servo_ids` | 编码方案和固定舵机顺序 |
| `fps` | 训练/关键动作频率，当前为1/0.15秒 |
| `hard_lower/upper` | 硬边界 |
| `soft_lower/upper` | 运行/训练编码使用的软边界 |
| `max_step` | 每个关键动作步允许的变化量 |
| `epsilon` | 数值比较容差 |

codec SHA256按文件字节计算，单纯重新格式化JSON也会改变哈希。客户端、服务端及模型绑定资产必须一致。改变动作周期、软限位、舵机顺序或编码后，不能继续假设旧checkpoint仍适配。

SmolVLA checkpoint需要 `model_contract.json`，检查schema_version=1、input_profile=`rgb_state_v1`、camera_mapping=`single_camera_camera1`和action_representation=`dynamic_bounded_relative_v1`，并从checkpoint读取 `action_codec.json`。训练worker会复制这些资产。π0.5通过自定义输入输出变换和归一化资产适配；当前runner明确要求72行，不应只修改服务配置来改变其horizon。

## 5. 路由器、worker和RPC

路由器运行于轻量环境，GPU模型在相应框架环境的worker进程中加载。切换模型会先停止旧worker，再启动新worker，状态写入 `run/workers/`，日志位于 `logs/worker-<model_id>.log`。

| RPC | 职责 |
|---|---|
| `Health` | 活跃模型、就绪状态、设备、版本、codec哈希 |
| `ListModels` | 模型列表、默认和当前活跃模型 |
| `GetModelStatus` | 查询指定模型状态 |
| `Predict` | 校验观测、选择模型、推理并返回动作块 |
| `ReleaseSession` | 释放客户端执行会话对模型的占用 |

每次Predict指定 `model_id`。切换过程返回 `MODEL_LOADING`，客户端等待ready后重新采集观测；不能在长时间加载后执行旧观测对应的预测。会话通过 `session_id`锁定模型，另一个会话切换会得到 `MODEL_SWITCH_LOCKED`；模型加载、会话占用和全局推理忙是不同状态。

请求图像为uint8 RGB/BGR buffer加宽高、步长；状态恰好六个有限值；任务非空；camera_mapping固定。响应包含原sequence/session/model标识、行列、展平的N×6绝对动作、服务端耗时及codec哈希。修改解析时同时覆盖缓冲区长度、行步长、非有限值和请求响应一致性。

服务端 `motion_allowed`由本次请求的 `request_execution`、服务级许可和模型 `can_execute`共同决定。配置中的 `allow_motion`与 `--allow-motion`在启动器中是“或”的关系。模型状态 `trained_dry_run`只能推理；`validated`且 `robot_compatible=true`才具备模型级执行资格。服务端自己不发布ROS动作，也不判断木块是否抓取成功。

gRPC仅绑定IPv4 loopback，用SSH隧道提供传输加密和身份认证；当前不是公网TLS服务。协议中的客户端capture时间不能与服务端UTC时间直接做差计算网络延迟。

## 6. 配置与实验复现

配置路径解析以仓库根目录为准；CLI输入的image、model等普通相对路径经各脚本进入项目后按当前目录处理。配置应存参数，不存密码、PID或动态worker状态。

SmolVLA训练快照保存在 `logs/training/<run_name>/`，包含训练/数据/codec配置和命令；checkpoint保存模型契约。π0.5实验通过experiment和step区分，归一化统计位于相应OpenPI资产路径。复现实验时同时记录本仓库提交、两个框架提交、lock文件、原始包SHA256、数据划分、训练参数、GPU选择和checkpoint，不只保留active软链接。

## 7. 测试

当前测试位于 `tests/unit/`，使用unittest。部分测试需要LeRobot、OpenCV及pyarrow，所以使用LeRobot环境运行，而不是只装了路由器依赖的 `.venv`：

```bash
cd /root/robot-vla
export PYTHONPATH="$PWD/src"
../lerobot/.venv/bin/python -m unittest discover -s tests/unit -v
```

以上路径适用于默认同级布局；自定义框架路径时使用 `config/frameworks.json`对应的解释器。不要使用 `discover -s tests`替代上述命令：当前子目录布局可能导致发现0项测试，而进程仍返回成功。

数据测试使用合成episode，覆盖哈希、时序、v7严格读取、终态保持和划分；部署测试覆盖相对路径和GPU传递。测试通过不代表GPU权重已加载，更不代表真实机器人抓取已验收。

最小集成验证顺序：安装检查 → 模型资产/框架自检 → 启动仅推理服务 → 客户端SSH检查 → 返回一次动作链 → 上传包校验。需要访问GPU或启动服务时选择空闲设备；仅修改文档不需要启动服务来验证。

## 8. 协议及新模型开发

协议源为 `src/proto/remote_inference.proto`，与客户端 `src/remote_rpc/proto/remote_inference.proto`保持一致。当前gRPC绑定使用grpcio-tools 1.73.1生成；在安装了匹配工具的开发环境执行protoc，输出到 `src/proto/`，再将生成的绝对pb2导入改为包内相对导入。可参考客户端的 `tools/generate_proto.py`；它只生成客户端目标目录，不会自动更新服务器。

```bash
python -m grpc_tools.protoc \
  -I src/proto \
  --python_out=src/proto \
  --grpc_python_out=src/proto \
  src/proto/remote_inference.proto
```

修改协议后提交定义和生成代码，校验两端导入及RPC行为。不要为了生成绑定而把JAX、PyTorch及工具依赖全部混入路由器环境。

新增模型应添加框架环境、runner及输入输出适配，注册到 `models.json`并补充worker分发。首次注册使用 `base_only`或 `trained_dry_run`等合适状态；完成当前机器人数据及动作契约验证后再开放执行。保留SmolVLA与π0.5现有模型ID和单RGB/六维输出，不静默补充虚构关节或把原生模型动作直接透传给客户端。

## 9. Git与交付

跟踪代码、配置、lock文件、脚本、测试及文档；不跟踪数据、权重、虚拟环境、日志和运行文件。`config/`当前受Git跟踪，本机修改应明确保留或提交，升级遇到冲突需要处理，安装器不会重建一套配置覆盖它。

提交前按改动风险检查对应测试、配置和README命令；若改动框架版本或依赖，做新的环境安装验证。临时脚本、源码备份和验证报告放在项目外的临时目录，保持正式目录可通过Git克隆和 `./install.sh`恢复。克隆不包含模型和数据资产，交付时另行准备。
