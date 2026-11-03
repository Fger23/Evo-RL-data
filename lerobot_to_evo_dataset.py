import argparse
import logging
import shutil
from pathlib import Path
from typing import List

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
# from lerobot.datasets.lerobot_dataset import (
#     LeRobotDataset,
#     LeRobotDatasetMetadata,
#     MultiLeRobotDataset,
# )
from tqdm import tqdm
import einops
import numpy as np

# 配置日志 (包含文件名和行号)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def parse_image(image) -> np.ndarray:
    # print('image shape:', image.shape) # image shape: torch.Size([3, 480, 640])
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

def _ensure_human_inloop_compatible_features(
    dataset_features: dict[str, dict],
    *,
    action_feature_names: list[str],
) -> None:
    # Keep human-in-loop datasets schema-stable across teleop-only and policy-assisted phases so they can merge.
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_feature_names),),
        "names": action_feature_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }

def resolve_collector_policy_id(
    *,
    intervention_enabled: bool,
    is_intervention: bool,
    selected_from_policy: bool,
    policy_id: str,
    human_id: str,
) -> str:
    """Resolve frame-level `collector_policy_id` from control mode and source."""
    if intervention_enabled:
        return human_id if is_intervention else policy_id
    return policy_id if selected_from_policy else human_id

def lerobot_to_evo_datasets(input_dir: Path, output_dir: Path):
    """
    将LeRobot数据集转换为EVO-RL数据集
    
    """

    # 验证所有输入目录都存在
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    
    # 加载第一个数据集的元数据作为模板
    input_metadata = LeRobotDatasetMetadata("", root=input_dir)
    dataset_features = input_metadata.features.copy()
    action_names = dataset_features["action"]["names"]
    action_names = list(action_names)
    _ensure_human_inloop_compatible_features(dataset_features, action_feature_names=action_names)
    dataset_features["complementary_info.collector_policy_id"] = {
            "dtype": "string",
            "shape": (1,),
            "names": ["collector_policy_id"],
        }
    # logging.info(f"转换后的数据集的features: {dataset_features}")
    # exit()

    action_feature_names = dataset_features["action"]["names"]
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    
    # 创建新的EVO-RL数据集
    logging.info(f"创建EVO-RL后的数据集: {output_dir}")
    output_dataset = LeRobotDataset.create(
        repo_id=None,
        robot_type=input_metadata.robot_type,
        root=output_dir,
        fps=input_metadata.fps,
        use_videos=True,
        features=dataset_features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    
    
    total_episodes = 0


    # 开始转换数据
    is_intervention = 0.0
    intervention_state = 0.0
    intervention_enabled = False
    selected_from_policy = False
    collector_policy_id_policy = "policy"
    collector_policy_id_human = "human"
    # 加载当前数据集的元数据
    metadata = LeRobotDatasetMetadata("", root=input_dir)
    for episode_idx in tqdm(range(len(metadata.episodes))):
        # try:
        if True:
            # 加载原始episode数据
            dataset = LeRobotDataset("", input_dir, episodes=[episode_idx])
            
            # 遍历每个step并添加到合并的数据集
            for data_item in dataset:
                # 准备图像数据
                image_dict = {}
                for key, value in data_item.items():
                    if "observation.image" in key:
                        image_key = key.split(".")[-1]
                        image_dict[f"observation.images.{image_key}"] = parse_image(value)
                frame = {
                        **image_dict,
                        "observation.state": data_item["observation.state"],
                        "action": data_item["action"],
                        "task": data_item["task"],
                    }
                frame["complementary_info.policy_action"] = np.array([zero_policy_action[name] for name in action_feature_names], dtype=np.float32)
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
                frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                    intervention_enabled=intervention_enabled,
                    is_intervention=bool(is_intervention),
                    selected_from_policy=selected_from_policy,
                    policy_id=collector_policy_id_policy,
                    human_id=collector_policy_id_human,
                )
                # 添加帧数据
                output_dataset.add_frame(frame)
            
            # 保存当前episode
            # merged_dataset.save_episode()
            extra_episode_metadata = (
                    {"episode_success": "success"}
                )
            output_dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
            
        # except Exception as e:
        #     logging.error(f"合并数据集 {dataset_idx + 1} 的episode {episode_idx} 时出错: {e}")
        #     continue
    
    logging.info(f"数据集转换完成! 输出目录: {output_dir}")

def dualArm2oneArm_lerobot_datasets(input_dirs: List[Path], output_dir: Path, arm_name: str = "merged_dataset"):
    """
    把双臂数据集转成单臂
    
    Args:
        input_dirs: 输入数据集目录列表
        output_dir: 输出合并后的数据集目录
        arm_name: 删除的臂
    """
    if not input_dirs:
        raise ValueError("至少需要提供一个输入数据集目录")
    
    # 验证所有输入目录都存在
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
    
    logging.info(f"输入数据集目录列表: {input_dirs}")
    # 创建输出目录
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载第一个数据集的元数据作为模板
    logging.info(f"加载第一个数据集的元数据: {input_dirs[0]}")
    first_metadata = LeRobotDatasetMetadata("", root=input_dirs[0])

    # logging.info(f"first_metadata features: {first_metadata.features}")
    # logging.info(f"features type: {type(first_metadata.features)}")
    # ====================== 关键修改 ======================
    # 复制 features 并删除 observation.images.left_wrist
    new_features = first_metadata.features.copy()
    delete_image = f"observation.images.{arm_name}_wrist"
    if delete_image in new_features:
        del new_features[delete_image]
        logging.info(f"已删除字段：{delete_image}")

    if "action" in new_features:
        new_features["action"]["names"] = [
                # "right_shoulder_pan.pos",
                # "right_shoulder_lift.pos",
                # "right_elbow_flex.pos",
                # "right_wrist_flex.pos",
                # "right_wrist_roll.pos",
                # "right_gripper.pos"
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos"
            ]
        new_features["action"]["shape"] = (6,)
    if "observation.state" in new_features:
        new_features["observation.state"]["names"] = [
                # "right_shoulder_pan.pos",
                # "right_shoulder_lift.pos",
                # "right_elbow_flex.pos",
                # "right_wrist_flex.pos",
                # "right_wrist_roll.pos",
                # "right_gripper.pos"
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos"
            ]
        new_features["observation.state"]["shape"] = (6,)
    # ======================================================
    # logging.info(f"new_features: {new_features}")
    # exit()
    # new_features = first_metadata.features

    
    # 创建新的LeRobot数据集
    logging.info(f"创建合并后的数据集: {output_dir}")
    merged_dataset = LeRobotDataset.create(
        repo_id=None,
        robot_type=first_metadata.robot_type,
        root=output_dir,
        fps=first_metadata.fps,
        use_videos=True,
        features=new_features,
        image_writer_threads=2,
        image_writer_processes=1,
    )

    # merged_dataset.features["observation.state"].shape = (6,)
    # merged_dataset.features["action"].shape = (6,)
    
    total_episodes = 0
    for input_dir in input_dirs:
        # 加载当前数据集的元数据
        metadata = LeRobotDatasetMetadata("", root=input_dir)
        total_episodes += len(metadata.episodes)
    
    logging.info(f"开始合并 {len(input_dirs)} 个数据集，共 {total_episodes} 个episodes")
    
    # 合并所有数据集
    for dataset_idx, input_dir in enumerate(input_dirs):
        logging.info(f"处理数据集 {dataset_idx + 1}/{len(input_dirs)}: {input_dir}")
        
        # 加载当前数据集的元数据
        metadata = LeRobotDatasetMetadata("", root=input_dir)
        # delete_episodes = [11, 23, 25, 29, 34, 36, 38, 41, 46, 49, 53, 55, 58, 61, 140, 146, 147, 159, 162, 172, 177, 184, 189, 192, 193, 198, 199, 200]
        for episode_idx in tqdm(range(len(metadata.episodes)), desc=f"数据集 {dataset_idx + 1}"):
            # if episode_idx in delete_episodes:
            #     continue
            try:
            # if True:
                # 加载原始episode数据
                dataset = LeRobotDataset("", input_dir, episodes=[episode_idx])
                
                # 遍历每个step并添加到合并的数据集
                for data_item in dataset:
                    # 准备图像数据
                    image_dict = {}
                    for key, value in data_item.items():
                        if "observation.image" in key and key in new_features.keys():
                            image_key = key.split(".")[-1]
                            image_dict[f"observation.images.{image_key}"] = parse_image(value)
                    
                    # 添加帧数据
                    if arm_name == 'left':
                        state = data_item["observation.state"][6:].squeeze()
                        action = data_item["action"][6:].squeeze()
                    elif arm_name == 'right':
                        state = data_item["observation.state"][:6].squeeze()
                        action = data_item["action"][:6].squeeze()
                    else:
                        raise ValueError(f"删除的臂类型不对: {arm_name}， 只能删除left/right")
                    # state = np.asarray(data_item["observation.state"][6:], dtype=np.float32).reshape(6,-1)
                    # action = np.asarray(data_item["action"][6:], dtype=np.float32).reshape(6,-1)

                    # logging.info(f'observation.state: {data_item["observation.state"].shape}')
                    # logging.info(f'action shape: {data_item["action"].shape}')
                    # logging.info(f"state shape: {state.shape}")
                    # logging.info(f"action shape: {action.shape}")
                    # exit()
                    merged_dataset.add_frame(
                        {
                            **image_dict,
                            "observation.state": state,
                            "action": action,
                            "task": data_item["task"],
                        }
                        # task=data_item.get("task", "")
                    )
                
                # 保存当前episode
                merged_dataset.save_episode()
                
            except Exception as e:
                logging.error(f"合并数据集 {dataset_idx + 1} 的episode {episode_idx} 时出错: {e}")
                continue
    
    logging.info(f"数据集合并完成! 输出目录: {output_dir}")
    logging.info(f"合并了 {len(input_dirs)} 个数据集，共 {total_episodes} 个episodes")


def main():
    parser = argparse.ArgumentParser(description="合并多个LeRobot数据集")
    
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="输入LeRobot数据集")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="输出转换后的数据集目录")
    # parser.add_argument("--dataset-name", type=str, default="merged_dataset",
    #                    help="输出数据集名称")
    # parser.add_argument("--delete-arm", type=bool, default=False,
    #                    help="删除机械臂模式")
    # parser.add_argument("--arm-name", type=str, default="left",
    #                    help="输出数据集名称")
    
    args = parser.parse_args()
    
    # # 验证参数
    # if not args.input_dirs:
    #     raise ValueError("至少需要提供一个输入数据集目录")
    
    # if args.delete_arm:
    #     print(f"delete arm: {args.arm_name}, enter a key to continue...")
    #     inp = input()
    #     dualArm2oneArm_lerobot_datasets(
    #     input_dirs=args.input_dirs,
    #     output_dir=args.output_dir,
    #     arm_name=args.arm_name
    # )
    # else:
    #     print(f"merge datasets: {args.input_dirs}, enter a key to continue...")
    #     inp = input()
    #     # 执行合并
    #     lerobot_to_evo_datasets(
    #         input_dirs=args.input_dirs,
    #         output_dir=args.output_dir,
    #         dataset_name=args.dataset_name
    #     )

    lerobot_to_evo_datasets(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )



if __name__ == "__main__":
    main()

    '''
    用法示例:

    python lerobot_to_evo_dataset.py \
        --input-dir /nfs/lerobot/s101/datasets/aggregation/take_and_read_card_06160617/ \
        --output-dir /nfs/lerobot/s101/datasets/aggregation/evo-rl/take_and_read_card_06160617/
        
    '''