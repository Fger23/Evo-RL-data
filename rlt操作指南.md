# Windows + 天钡 RLT 操作指南（全程策略输出 S-G 7 帧）

Windows `192.168.41.1` 负责 BF16 VLA/RLT 推理、缓存与训练；天钡 `192.168.41.2` 负责观测、CAN 下发、人工接管和录制。按 1.1→1.8 完成首轮，再按 1.9 迭代。本指南使用当前仓库源码，两端先同步更新；服务端源码更新后需重启服务。

本版基础阶段为 **RTC VLA→S-G→下发**，RLT 阶段为 **RTC VLA→Actor→S-G→下发**，区间外也使用 S-G；Actor 仍接收未滤波 VLA 参考。所有采集保持 30 FPS、无损 PNG 和 `action_sg_window=7`，缓存使用 `--sg-window 7`，记录 `sg_stage=client_output`。旧无滤波头或旧“先 S-G 再 RLT”的头不能直接用于这组配置，需要重新构建缓存和训练，不能手改合同。

S-G 只处理策略输出的关节位置，不处理夹爪，人工接管仍沿用原遥操目标。7 帧窗口保留块首尾各 3 帧，5–6 帧回退到 5 帧窗口，少于 5 帧不滤波；“全程开启”指每个策略阶段都请求滤波，不是强行改变所有边缘帧。滤波发生在客户端裁掉过期动作之后；RTC 模型空间前缀仍沿用滤波前的 VLA/Actor 计划，不把 S-G 误当作姿态对齐或跨块插帧。

每次采集使用新名称。下例首批 `Remove_rtc_rollout_1`，下一批 `Remove_rtc_rlt_rollout_2`。首批 50 回合、后续 20 回合及训练步数是试验起点，不是效果保证。独立验证数据不得加入训练清单；当前训练器不自动划分训练/验证。

## 1.1 Windows：启动模型服务（PowerShell）

这一步在 Windows 执行，保持窗口运行。启动时把模型加载到 CPU，客户端通过握手选择 Windows 的 GPU，再开始天钡采集。显式调用虚拟环境 Python，不需要额外激活。

```powershell
Set-Location E:\ABYT\a0810\Evo-RL-data-InferenceTimeRTC

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

& E:\ABYT\a0810\venvs\pi05-server\Scripts\python.exe -m lerobot.inference_time_rtc.policy_server `
  --host=0.0.0.0 `
  --port=8110 `
  --policy_type=pi05 `
  --pretrained_name_or_path=E:/ABYT/a0810/models/Remove1112_1_7_pi05/pretrained_model `
  --checkpoint_root=E:/ABYT/a0810/models `
  --tokenizer_name_or_path=E:/ABYT/a0810/models/paligemma-3b-pt-224 `
  --policy_dtype=bfloat16 `
  --compile_model=false `
  --warmup_on_connect=false
```

看到模型在 CPU 加载完成并监听 `8110` 后执行下一节。这里不开启 compile 或握手预热，首次推理仍可能较慢；`rpc_timeout_s` 只控制等待，不增加队列余量。两端链路与 Windows 8110 入站规则检查见 [操作指南](操作指南.md) §1.9.3。

## 1.2 天钡：基础 VLA 推理并采集首批数据（Bash）

下面会连接主从臂和相机，首轮关闭 RLT、保留 S-G。每个用于训练的 episode 至少按 `r` 标记一段起止，真实成功按 `s`，失败按 `f`；`i` 切入/退出人工接管。`r` 只标记训练区间，不启用 Actor。未标注区间的数据不能直接构建本流程缓存。自然超时默认失败，不能把真实成功留给超时自动判断。

```bash
source /home/apex/anaconda3/etc/profile.d/conda.sh
conda activate /home/apex/anaconda3/envs/evo-rl
cd /home/apex/code/Evo-RL-data-InferenceTimeRTC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export no_grpc_proxy=192.168.41.1,127.0.0.1,localhost
export no_proxy=192.168.41.1,127.0.0.1,localhost

rtc_run="Remove_rtc_rollout_1"     # 每次新采集使用新的名称
rtc_capture="/home/apex/.cache/huggingface/lerobot/piper/$rtc_run"

python -m lerobot.inference_time_rtc.rollout \
  --server_address=192.168.41.1:8110 \
  --policy_type=pi05 \
  --pretrained_name_or_path=E:/ABYT/a0810/models/Remove1112_1_7_pi05/pretrained_model \
  --policy_device=cuda:0 \
  --rpc_timeout_s=1800 \
  --allow_late_splice=false \
  --actions_per_chunk=50 \
  --action_sg_window=7 \
  --robot.type=bi_piper_follower \
  --robot.left_arm_config.port=can3 \
  --robot.right_arm_config.port=can2 \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras="{fixed: {type: intelrealsense, serial_number_or_name: '263122074897', width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: '263122072536', width: 640, height: 480, fps: 30}}" \
  --robot.right_arm_config.cameras="{wrist: {type: intelrealsense, serial_number_or_name: '262722075213', width: 640, height: 480, fps: 30}}" \
  --teleop.type=bi_piper_leader \
  --teleop.left_arm_config.port=can1 \
  --teleop.right_arm_config.port=can0 \
  --teleop.id=my_bi_piper_leader \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id="piper/$rtc_run" \
  --dataset.root="$rtc_capture" \
  --dataset.single_task='Remove the card from the slot and insert it into the card reader.' \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=240 \
  --dataset.reset_time_s=60 \
  --dataset.storage=pkl \
  --dataset.push_to_hub=false \
  --rtc.enabled=true \
  --rtc.execution_horizon=20 \
  --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc.queue_threshold=30 \
  --rtc.auto_delay=false \
  --rtc.inference_delay=16 \
  --rtc.observation_image_codec=lossless_png \
  --rtc.latency_window=20 \
  --recovery.enabled=true \
  --rlt.enabled=false
```
## 1.3 将数据传回

### 天钡：采集正常结束后，检查并转换 PKL

只改 `rtc_run` 一行。保留整个原始目录中的 `capture.pkl` 与 `episodes/`；该转换命令不会启动机械臂。

```bash
source /home/apex/anaconda3/etc/profile.d/conda.sh
conda activate /home/apex/anaconda3/envs/evo-rl
cd /home/apex/code/Evo-RL-data-InferenceTimeRTC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

rtc_run="Remove_rtc_rollout_1"

python convert_pkl_dataset.py \
  --input "/home/apex/.cache/huggingface/lerobot/piper/$rtc_run/capture.pkl" \
  --validate-only && \
python convert_pkl_dataset.py \
  --input "/home/apex/.cache/huggingface/lerobot/piper/$rtc_run/capture.pkl" \
  --output "/home/apex/.cache/huggingface/lerobot/piper/${rtc_run}-dataset" \
  --no-upload
```

### Windows：复制转换后数据，设置本轮训练目录（PowerShell）

先正常结束天钡采集，再在 Windows 服务窗口按 `Ctrl+C` 停止服务，释放显存。另开 PowerShell 执行下面整段，只有 `rtcRun` 需要按本批名称修改。`scp` 按提示输入天钡 SSH 密码。本节后的缓存、训练命令沿用这个终端的变量；换终端时重新执行环境和变量设置。

```powershell
Set-Location E:\ABYT\a0810\Evo-RL-data-InferenceTimeRTC
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$py = "E:/ABYT/a0810/venvs/pi05-server/Scripts/python.exe"

$rtcRun = "Remove_rtc_rollout_1"
$dataParent = "E:/ABYT/a0810/datasets"
$rtcDataset = "$dataParent/${rtcRun}-dataset"
$rltRoot = "E:/ABYT/a0810/models/rtc_rlt_Remove1112_1_7/${rtcRun}_output_sg7"
$rtcModel = "E:/ABYT/a0810/models/Remove1112_1_7_pi05/pretrained_model"
$rtcTask = "Remove the card from the slot and insert it into the card reader."
New-Item -ItemType Directory -Force -Path @($dataParent, $rltRoot) | Out-Null

scp -r "apex@192.168.41.2:/home/apex/.cache/huggingface/lerobot/piper/${rtcRun}-dataset" "$dataParent/"
if ($LASTEXITCODE -ne 0) { throw "数据复制失败" }
if (-not (Test-Path "$rtcDataset/meta/info.json")) { throw "数据目录不完整" }
```

## 1.4 Windows：生成数据清单和训练缓存（PowerShell）

### 首轮数据清单

首次训练执行本段；后续累计数据时使用 1.9 的清单，不要重新运行本段覆盖它。保存为无 BOM UTF-8 JSON 数组。

```powershell
$specPath = "$rltRoot/datasets_output_sg7.json"
if (Test-Path $specPath) { throw "清单已存在，请确认本轮名称或直接使用已有清单" }
$datasetSpec = @(@{repo_id="piper/$rtcRun"; root=$rtcDataset; source="rollout"})
$specJson = ConvertTo-Json -InputObject $datasetSpec -Depth 4
[IO.File]::WriteAllText($specPath, $specJson, [Text.UTF8Encoding]::new($false))
```

### 构建完整缓存

```powershell
& $py -m lerobot.inference_time_rtc.rlt.cache `
  --base-model "$rtcModel" `
  --dataset-spec "$rltRoot/datasets_output_sg7.json" `
  --output "$rltRoot/transitions_output_sg7.pt" `
  --device cuda:0 `
  --dtype bfloat16 `
  --task "$rtcTask" `
  --fps 30 `
  --chunk-size 50 `
  --sg-window 7 `
  --execution-horizon 20 `
  --inference-delay 16 `
  --queue-threshold 30 `
  --max-guidance-weight 10 `
  --prefix-attention-schedule EXP `
  --token-pool-size 64

if ($LASTEXITCODE -ne 0) { throw "缓存构建失败，不要继续训练" }
```

`--sg-window 7` 描述 Actor 后的客户端输出滤波，不把 VLA 参考先滤波。Critic 使用实录实际下发目标；训练对 Actor 与 TD 目标动作加入可微 S-G 后再评估 Q，BC 仍约束原始 VLA 参考。缓存里的模型路径、processor 指纹、动作顺序、FPS 和 RTC 参数必须与部署一致。必须在 Windows 最终模型路径构建缓存；可以把这个 Windows 缓存上传 H200 训练，再把 Actor 复制回来，但不能直接拿原 `/nfs/...` 合同的头改路径使用。

缓存输出不能覆盖或追加；重建时换新实验目录。首次小跑可将输出改为 `cache_smoke_output_sg7.pt` 并增加 `--max-episodes 1`，随后仍需生成完整缓存。离线参考和 S-G 按有效完整块重放，不精确重建网络抖动、过期裁剪和恢复事件。

## 1.5 Windows：训练编码器／解码器和 Actor／Critic（PowerShell）

在 1.3–1.4 同一训练终端继续执行。本节不启动模型服务或机械臂；不要同时运行服务占用 GPU。若显存不足先减小 batch-size，不为省显存改变 FPS、chunk 或动作维度。

```powershell
Set-Location E:\ABYT\a0810\Evo-RL-data-InferenceTimeRTC

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$py = "E:/ABYT/a0810/venvs/pi05-server/Scripts/python.exe"

$rtcRun = "Remove_rtc_rollout_1"
$rltRoot = "E:/ABYT/a0810/models/rtc_rlt_Remove1112_1_7/${rtcRun}_output_sg7"

if (-not (Test-Path "$rltRoot/transitions_output_sg7.pt")) {
    throw "没有找到训练缓存，请先完成 rlt.cache"
}

# 第一阶段：训练 Token 编码器和解码器
& $py -m lerobot.inference_time_rtc.rlt.training token `
  --cache "$rltRoot/transitions_output_sg7.pt" `
  --output "$rltRoot/token_output_sg7.pt" `
  --device cuda:0 `
  --steps 1000 `
  --batch-size 16 `
  --learning-rate 0.0001 `
  --log-every 100 `
  --save-every 500

if ($LASTEXITCODE -ne 0) {
    throw "Token 训练失败，请检查错误后再继续"
}

# 第二阶段：冻结 Token 编码器，训练 Actor 和双 Critic
& $py -m lerobot.inference_time_rtc.rlt.training ac `
  --cache "$rltRoot/transitions_output_sg7.pt" `
  --token-checkpoint "$rltRoot/token_output_sg7.pt" `
  --output "$rltRoot/actor_output_sg7.pt" `
  --device cuda:0 `
  --steps 10000 `
  --batch-size 32 `
  --actor-lr 0.0003 `
  --critic-lr 0.0003 `
  --gamma 0.99 `
  --beta 1.0 `
  --log-every 100 `
  --save-every 1000

if ($LASTEXITCODE -ne 0) {
    throw "Actor/Critic 训练失败，请检查错误"
}

Write-Host "部署权重：$rltRoot/actor_output_sg7.pt"
```

Token 阶段训练编码器和解码器；AC 阶段冻结 Token 编码器、交替更新 Actor 与双 Critic，VLA 始终冻结。部署用 `actor_output_sg7.pt`，不用 token 文件。训练损失不代表实机成功率。

## 1.6 Windows：不连接机械臂检查训练产物（PowerShell）

在训练终端执行，CPU 加载头并试跑一条缓存，不加载 VLA 或连接 8110。

```powershell
@'
import sys
import torch
from lerobot.inference_time_rtc.rlt.networks import RLTHeadBundle

cache = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
head = RLTHeadBundle.load(sys.argv[2], device="cpu", require_actor=True)
assert head.contract == cache["metadata"]["contract"]
assert head.contract["sg_window"] == 7 and head.contract["sg_stage"] == "client_output"
data = cache["tensors"]
with torch.inference_mode():
    output = head.predict_chunk(
        data["vla_tokens"][:1].float(), data["proprio"][:1].float(),
        data["reference_actions"][:1].float(), data["protected_steps"][:1],
        action_mask=data["rlt_action_mask"][:1],
    )
assert torch.isfinite(output).all()
assert output.shape == (1, head.config.chunk_size, head.config.action_dim)
print("stage:", head.trained_stage, "updates:", head.training_updates)
print("Output-SG7 head check OK")
'@ | & $py - "$rltRoot/transitions_output_sg7.pt" "$rltRoot/actor_output_sg7.pt"
if ($LASTEXITCODE -ne 0) { throw "权重检查失败，不要部署" }
```

## 1.7 Windows：重新启动模型服务（PowerShell）

保持此窗口运行；Actor 路径由天钡握手指定。若尚有旧服务先正常结束，避免端口或显存冲突。

```powershell
Set-Location E:\ABYT\a0810\Evo-RL-data-InferenceTimeRTC

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

& E:\ABYT\a0810\venvs\pi05-server\Scripts\python.exe -m lerobot.inference_time_rtc.policy_server `
  --host=0.0.0.0 `
  --port=8110 `
  --policy_type=pi05 `
  --checkpoint_root=E:/ABYT/a0810/models `
  --tokenizer_name_or_path=E:/ABYT/a0810/models/paligemma-3b-pt-224 `
  --policy_dtype=bfloat16 `
  --compile_model=false `
  --warmup_on_connect=false
```
## 1.8 天钡：使用 Actor 推理并采集下一批（Bash）

下面继续保留 S-G 7 帧和 PNG。Actor 的步数区间按实际首轮标注统计填写；这只控制 Actor 启用范围，不会自动生成下一批训练标注。新一批每回合仍按 `r` 标区间、`s/f` 标结果，`i` 接管不变。第一次验证可先将 20 回合改为 5 回合。`rlt_head` 是 Windows 文件路径，天钡不需要复制模型。
```bash
source /home/apex/anaconda3/etc/profile.d/conda.sh
conda activate /home/apex/anaconda3/envs/evo-rl
cd /home/apex/code/Evo-RL-data-InferenceTimeRTC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export no_grpc_proxy=192.168.41.1,127.0.0.1,localhost
export no_proxy=192.168.41.1,127.0.0.1,localhost

rtc_run="Remove_rtc_rlt_rollout_2"  # 每批使用新的名称
rtc_capture="/home/apex/.cache/huggingface/lerobot/piper/$rtc_run"

rlt_head="E:/ABYT/a0810/models/rtc_rlt_Remove1112_1_7/Remove_rtc_rollout_1_output_sg7/actor_output_sg7.pt"

read -r -p "RLT 起始步数: " rlt_start
read -r -p "RLT 结束步数: " rlt_end

python -m lerobot.inference_time_rtc.rollout \
  --server_address=192.168.41.1:8110 \
  --policy_type=pi05 \
  --pretrained_name_or_path=E:/ABYT/a0810/models/Remove1112_1_7_pi05/pretrained_model \
  --policy_device=cuda:0 \
  --rpc_timeout_s=1800 \
  --allow_late_splice=false \
  --actions_per_chunk=50 \
  --action_sg_window=7 \
  --robot.type=bi_piper_follower \
  --robot.left_arm_config.port=can3 \
  --robot.right_arm_config.port=can2 \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras="{fixed: {type: intelrealsense, serial_number_or_name: '263122074897', width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: '263122072536', width: 640, height: 480, fps: 30}}" \
  --robot.right_arm_config.cameras="{wrist: {type: intelrealsense, serial_number_or_name: '262722075213', width: 640, height: 480, fps: 30}}" \
  --teleop.type=bi_piper_leader \
  --teleop.left_arm_config.port=can1 \
  --teleop.right_arm_config.port=can0 \
  --teleop.id=my_bi_piper_leader \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id="piper/$rtc_run" \
  --dataset.root="$rtc_capture" \
  --dataset.single_task='Remove the card from the slot and insert it into the card reader.' \
  --dataset.fps=30 \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=240 \
  --dataset.reset_time_s=60 \
  --dataset.storage=pkl \
  --dataset.push_to_hub=false \
  --rtc.enabled=true \
  --rtc.execution_horizon=20 \
  --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc.queue_threshold=30 \
  --rtc.auto_delay=false \
  --rtc.inference_delay=16 \
  --rtc.observation_image_codec=lossless_png \
  --rtc.latency_window=20 \
  --recovery.enabled=true \
  --rlt.enabled=true \
  --rlt.checkpoint="$rlt_head" \
  --rlt.start_step="$rlt_start" \
  --rlt.end_step="$rlt_end"
```

服务端回包仍是未做 S-G 的物理动作，天钡接入可执行块后再滤波一次。`filtered_action_chunk` 的 `sg_window=7`（短块可能为 5 或 0）和逐帧 `sent_action` 用于核对实际下发；`rlt_applied` 标记 Actor 来源，不表示该帧没有后续滤波。S-G 的窗口可以跨 Actor 掩码边缘，最终目标不等于严格逐行保留未滤波 VLA。它不能保证消除跨块跳变，也不是关节速度/幅度限制器。

## 1.9 第二轮及以后：累计数据，只继续 AC

先正常结束 1.8 的采集，按 1.3 转换并复制本批数据。将 1.3 两端的名称改为 `Remove_rtc_rlt_rollout_2`，Windows 会建立对应的 `_output_sg7` 新目录。停止 Windows 服务后，在该训练终端执行下面的累计清单命令，保留上一轮目录，不覆盖旧头。

```powershell
$previousRoot = "E:/ABYT/a0810/models/rtc_rlt_Remove1112_1_7/Remove_rtc_rollout_1_output_sg7"
$previousActor = "$previousRoot/actor_output_sg7.pt"
$specPath = "$rltRoot/datasets_output_sg7.json"
if (-not (Test-Path $previousActor)) { throw "上一轮 Actor 不存在" }
if (Test-Path $specPath) { throw "本轮清单已存在，请确认使用了新采集名称" }
$datasetSpec = @(Get-Content -Raw -Encoding UTF8 "$previousRoot/datasets_output_sg7.json" | ConvertFrom-Json)
if (@($datasetSpec | Where-Object { $_.root -eq $rtcDataset }).Count -gt 0) {
    throw "新增数据已在上一轮清单中"
}
$datasetSpec += @{repo_id="piper/$rtcRun"; root=$rtcDataset; source="rollout"}
$specJson = ConvertTo-Json -InputObject $datasetSpec -Depth 4
[IO.File]::WriteAllText($specPath, $specJson, [Text.UTF8Encoding]::new($false))
```

再执行 **1.4 的“构建完整缓存”代码块**，不要执行“首轮数据清单”代码块。新缓存包含旧训练集和新增训练集；独立验证集不加入。缓存成功后不运行 1.5 的 token 命令，直接继续 AC：

```powershell
& $py -m lerobot.inference_time_rtc.rlt.training ac `
  --cache "$rltRoot/transitions_output_sg7.pt" `
  --token-checkpoint "$previousActor" `
  --output "$rltRoot/actor_output_sg7.pt" `
  --device cuda:0 `
  --steps 5000 `
  --batch-size 32 `
  --actor-lr 0.0003 `
  --critic-lr 0.0003 `
  --gamma 0.99 `
  --beta 1.0 `
  --log-every 100 `
  --save-every 1000
if ($LASTEXITCODE -ne 0) { throw "本轮 AC 训练失败" }
Write-Host "本轮部署权重：$rltRoot/actor_output_sg7.pt"
```

`--steps` 是本轮追加更新数。继承网络权重，Token 冻结；优化器和 target critic 重新建立，不是精确断点恢复。保持 VLA、processor、动作顺序、FPS、S-G 和 RTC 配置一致才能继续旧权重。

再执行 1.6 检查、1.7 启动服务；在 1.8 更换新 Actor 路径、使用第三批新名称。只部署通过比较的版本，分别记录无人接管成功、接管次数、耗时、队列恢复和明显关节跳变；损失下降不等于实机改善。

## 1.10 退回旧 Actor 或基础 RTC

先正常结束当前客户端。退回同配置的旧 Actor 时，在 1.8 使用保留的旧 `_output_sg7/actor_output_sg7.pt` 路径及新的采集名称。退回基础 RTC 时运行 1.2，使用新的 `rtc_run`，保持 `--rlt.enabled=false`、`--action_sg_window=7` 和 PNG。单纯换头或关闭 RLT 时，更新后的 Windows 服务可继续运行；修改服务端源码后需重启。

本次仅实现代码、整理指南并进行无硬件检查；没有自动执行这里的采集、模型训练或实机动作。旧原始指南和其他文档已备份在本地工作目录。GitHub、Windows/天钡运行进程及远端仓库不会因本地文件修改自动更新。
