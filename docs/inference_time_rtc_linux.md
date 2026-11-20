# Linux InferenceTimeRTC 远程推理

**当前已报告 RTC 抖动与撞桌，先急停或断使能并停止 RTC 实机运行。本文保留启动入口及历史模拟时延配置，不作为当前恢复实机的安全推荐。请先阅读 [RTC 调参与异常排查指南](rtc_tuning_guide.md)。**

本模式使用独立入口 `lerobot.inference_time_rtc`。服务端启动时将模型加载到 CPU，客户端连接时下发服务端显卡选择和全部 RTC 配置；模型迁移成功后才连接机器人。原有 `lerobot.async_inference.policy_server`、普通客户端、诊断客户端和录制客户端保持原样。

## 1. 环境和模型

两端使用同一版本的本仓库。服务端需要已经能运行 π0.5 的 PyTorch/CUDA、Transformers 和 gRPC 环境；客户端需要本仓库、gRPC、PiPER SDK、SocketCAN 和 RealSense 依赖，无需模型权重或 CUDA 显卡。

当前部署目录为服务器 `/home/baiyuntian/baiyuntian/Evo-RL-data-InferenceTimeRTC` 和天钡 `/home/apex/code/Evo-RL-data-InferenceTimeRTC`。下方命令先初始化 Conda 并显式激活已验证的 `evo-rl` 环境，再通过进程级 `PYTHONPATH` 加载当前仓库，使用 `python` 启动程序。全新环境先按照仓库原有安装说明安装模型/硬件依赖。

当前允许 π0.5、π0 和 SmolVLA 的原生 RTC 接口；首版以 π0.5 + 双臂 PiPER 为主要使用场景。EVO1、ACT 等不在新模式支持列表中，继续使用其现有推理入口。

## 2. Linux 服务端启动模型

以下使用原操作指南中的 Linux 模型路径与服务端地址。模型路径、GPU 编号、IP 按实际部署调整。

```bash
source /nfs/zxlong10/anaconda3/etc/profile.d/conda.sh
conda activate /nfs/zxlong10/anaconda3/envs/evo-rl
cd /home/baiyuntian/baiyuntian/Evo-RL-data-InferenceTimeRTC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m lerobot.inference_time_rtc.policy_server \
  --host=0.0.0.0 \
  --port=8100 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/nfs/lerobot/piper/models/evo/policy/straighten1110_to_1111_pi05/checkpoints/last/pretrained_model
```

终端出现 `InferenceTimeRTC model loaded on cpu; listening on ...` 表示模型已在 CPU 加载完成。保持该终端运行，再启动客户端选择 GPU。模型在连接时迁移到所选设备，首次模型前向预热仍发生在收到第一条观测之后。

服务端命令无需指定 GPU，不要在命令前固定 `CUDA_VISIBLE_DEVICES=0`。显卡由下面的客户端 `--policy_device=cuda:N` 指定，使用服务端进程的可见编号；如果服务端启动 shell 已设置 `CUDA_VISIBLE_DEVICES`，编号遵从其映射。

模型目录应包含模型配置、权重，以及训练导出的 preprocessor/postprocessor 配置与统计文件。精度沿用 checkpoint 配置。该入口在内存中关闭模型编译与梯度检查点，不改写磁盘上的 checkpoint。

如果 checkpoint 中的 tokenizer 路径来自 Windows 或另一台机器，在服务端命令末尾增加：

```bash
  --tokenizer_name_or_path=/actual/linux/path/paligemma-3b-pt-224
```

这是服务端本地可读取的 tokenizer 目录。也可使用当前环境能访问的 Hugging Face tokenizer ID。RTC 算法参数不放在服务端命令中。

## 3. Linux 客户端开始推理

本例沿用原指南中的双臂 CAN 口、三路相机编号和整钞任务。先确认这些参数与当前设备相符。CAN 配置、标定和相机准备沿用原来的流程。

```bash
source /home/apex/anaconda3/etc/profile.d/conda.sh
conda activate /home/apex/anaconda3/envs/evo-rl
cd /home/apex/code/Evo-RL-data-InferenceTimeRTC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export no_grpc_proxy=10.61.5.1,127.0.0.1,localhost
export no_proxy=10.61.5.1,127.0.0.1,localhost

python -m lerobot.inference_time_rtc.robot_client \
  --server_address=10.61.5.1:8100 \
  --policy_device=cuda:0 \
  --robot.type=bi_piper_follower \
  --robot.left_arm_config.port=can3 \
  --robot.right_arm_config.port=can2 \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.cameras="{fixed: {type: intelrealsense, serial_number_or_name: '263122074897', width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: '263122072536', width: 640, height: 480, fps: 30}}" \
  --robot.right_arm_config.cameras="{wrist: {type: intelrealsense, serial_number_or_name: '262722075213', width: 640, height: 480, fps: 30}}" \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --task="Pick up the banknotes from the desk, straighten them neatly and hand them over to the customer." \
  --fps=30 \
  --actions_per_chunk=50 \
  --rtc.enabled=true \
  --rtc.execution_horizon=30 \
  --rtc.max_guidance_weight=1.0 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc.queue_threshold=35 \
  --rtc.auto_delay=false \
  --rtc.inference_delay=22 \
  --rtc.latency_window=20
```

`require_calibration=false` 沿用原部署的设置；若你的机器人需要标定，应使用实际标定配置。任务描述也应与训练时一致。

模型路径与模型类型由服务端确定，GPU 由客户端 `policy_device` 确定。例如 `--policy_device=cuda:2` 选择服务端第 2 号可见 GPU，天钡本身无需 CUDA。省略此参数时通常选择服务端 `cuda:0`；旧服务端显式传入的预加载设备也可继续作为默认设备。可省略 `actions_per_chunk`，自动使用模型完整 chunk 长度。

显卡选择会在连接机器人前完成校验和分配；GPU 不存在或显存不足时客户端直接报错。单个服务端仍只接受一个客户端会话。先停止旧客户端，再启动指定另一 GPU 的新客户端时，模型经 CPU 迁移并清理旧 GPU 缓存；同卡重连复用权重。

当前使用 RTC 协议 v3，包含输入梯度引导修正，两端需要一起更新并重启。其他推理入口和 protobuf 定义未改变。当前应保持实机停机，先完成离线验证。

两个 CLI 自动把 INFO 日志保存到当前目录的 `logs/rtc_client_*.log` 或 `logs/rtc_server_*.log`，同时输出到终端，启动时打印完整路径。可通过 `--log_file=实际路径` 指定。旧版本没有自动保存文件，而且部分依赖可能隐藏 INFO 输出；旧日志不能追补。

按 Ctrl+C 停止客户端，会停止命令循环、取消未完成 RPC、释放 RTC 会话并断开机器人；服务端仍保留模型，可以接收下一个客户端。断开时是否禁用机械臂电机沿用机器人配置的 `disable_on_disconnect` 行为。

## 4. 客户端 RTC 参数

以下表格列出代码默认值。上方固定延迟 22、引导截止 30、请求阈值 35 的组合只做过模拟时延与队列验证；报告撞桌后不再作为实机推荐值。应按调参指南先核验实现与轨迹。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `rtc.enabled` | `true` | 本入口必须开启；其他模式使用原入口 |
| `rtc.execution_horizon` | `20` | 传入原生 RTC 的前缀权重截止位置，单位为控制步 |
| `rtc.max_guidance_weight` | `1.0` | RTC 引导权重上限 |
| `rtc.prefix_attention_schedule` | `EXP` | `EXP`、`LINEAR`、`ONES`、`ZEROS`，大小写如表所示 |
| `rtc.queue_threshold` | `30` | 剩余动作不多于该步数时，开始采集观测并请求下一块 |
| `rtc.auto_delay` | `true` | 自动估计完整观测到动作返回的延迟 |
| `rtc.inference_delay` | `10` | 自动模式的初始延迟估计；固定模式下每次使用该值 |
| `rtc.latency_window` | `20` | 自动估计最近多少次请求的 95% 分位延迟 |
| `rtc.debug` | `false` | 开启已有 RTC 模型侧调试记录 |
| `rtc.debug_maxlen` | `100` | 模型侧调试记录上限 |

约束：`0 <= inference_delay < execution_horizon <= queue_threshold < actions_per_chunk <= 模型chunk_size`。首次基础模型预热不计入延迟窗口。后续估计包含相机读取、序列化、网络传输、模型处理与客户端反序列化；不依赖两端时钟同步。

历史时延测试采用 `--rtc.auto_delay=false --rtc.inference_delay=22`。这些步数都以客户端 `--fps` 为单位，不是毫秒。固定模式下 `latency_window` 不参与计算；设为自动模式后，`inference_delay` 仅作为初始估计，不会持续保留 22 步余量。

自动测量结果可能大于初始估计，还必须检查实际旧前缀的长度。当前客户端在实际延迟达到有效截止 `min(execution_horizon, prefix_steps)` 时拒绝换块并中止，不再继续接入缺少直接前缀约束的未来动作。应停机后离线检查时延、吞吐和前缀预算，不能只增大 horizon、queue_threshold 或任意降低频率。RTC 不会让模型本身变快；不要直接将其他硬件测得的延迟当作本机参数。

## 5. 运行行为与排查

- 观测采集和网络请求在后台线程执行，动作执行有独立的客户端定时循环。
- 模型空间原始动作与后处理执行动作成对保留。后续请求把未消费的原始动作传回服务端进行 RTC 引导，不做旧模式的动作平均。
- 同时只允许一个未完成推理请求。响应包含请求编号和观测采集起点，旧响应不能覆盖新动作块。
- 第一块到达后才启动动作时间轴；后续切换按已过去的控制步丢弃过期动作。控制循环错过时限时跳过相应步，不突发补发旧指令。
- 首块到达前可以等待。动作时间轴启动后，队列耗尽会暂停下发。客户端继续采集观测并记录推理结果，后续响应不会自动恢复发令。暂停下发时，机械臂可能仍在追踪最后一次下发的目标。
- 每次换块记录 `max_abs_overlap_delta` 和关节名，比较同一未来时刻的新旧目标；这不是实际关节运动幅度，也不是安全阈值。详细分析见 [本次跳变日志分析](rtc_jump_analysis_20261118.md)。
- `estimated_delay` 是传给模型的预测延迟；`actual_delay` 是接收时实际丢弃的控制步数；`queue` 是切换后剩余动作数。
- 如需限定本次运行时间，可在客户端添加 `--duration_s=60`，计时包含首次观测预热。
- 服务端默认端口 `8100`，原模式仍可使用 `8099`。两者的客户端不能混连；新客户端会先检查协议，再连接机器人。
- 单个 RTC 服务端同时接收一个客户端。异常掉线后若会话没有正常释放，默认空闲 120 秒可重新占用，也可重启服务端；这是服务端的 `session_timeout_s` 参数。
- 新模式没有添加身份认证或 TLS，沿用原系统的可信局域网部署方式。

## 6. 验证范围

新增测试覆盖真实 gRPC 消息传输、配置握手、原始动作与执行动作隔离、延迟丢弃、冷启动、队列耗尽、重复请求、会话占用与释放、模型异常和相机采集期间的动作执行。

```bash
python -m pytest tests/inference_time_rtc tests/policies/rtc
```

这些自动化测试使用模拟模型与机器人，不会操作机械臂。真实 Linux GPU、相机、CAN 总线以及任务效果仍需要部署后联调。

2026-11-18 在天钡到 `10.61.5.1` 的实际链路上做过真实 NFS 模型时延测试：H200、float32、10 次去噪、三路 640×480 模拟图像；正式客户端约 35 秒内取得 64 个预热后样本，端到端平均约 461 ms、P95 约 564 ms、P99 约 683 ms、最大约 825 ms，未出现队列耗尽。当时按 P99 换算并留余量测试了固定延迟 22 步。这不包含真实相机等待与机械臂运动，本次梯度修正后还需重新测量，不能沿用为安全推荐。
