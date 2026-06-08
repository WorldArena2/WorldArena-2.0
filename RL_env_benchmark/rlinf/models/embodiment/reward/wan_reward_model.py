import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from typing import Union, List
import numpy as np
# LIBERO_GOAL 任务描述字典 (10 个任务)

LIBERO_GOAL_TASKS = {
    0: "open the middle drawer of the cabinet",
    1: "open the top drawer and put the bowl inside",
    2: "push the plate to the front of the stove",
    3: "put the bowl on the plate",
    4: "put the bowl on the stove",
    5: "put the bowl on top of the cabinet",
    6: "put the cream cheese in the bowl",
    7: "put the wine bottle on the rack",
    8: "put the wine bottle on top of the cabinet",
    9: "turn on the stove"
}

LIBERO_10_TASKS = {
    0: "pick up the book and place it in the back compartment of the caddy",
    1: "put both moka pots on the stove",
    2: "put both the alphabet soup and the cream cheese box in the basket",
    3: "put both the alphabet soup and the tomato sauce in the basket",
    4: "put both the cream cheese box and the butter in the basket",
    5: "put the black bowl in the bottom drawer of the cabinet and close it",
    6: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    7: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    8: "put the yellow and white mug in the microwave and close it",
    9: "turn on the stove and put the moka pot on it"
}

LIBERO_SPATIAL_TASKS = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl from table center and place it on the plate",
    2: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    3: "pick up the black bowl next to the cookie box and place it on the plate",
    4: "pick up the black bowl next to the plate and place it on the plate",
    5: "pick up the black bowl next to the ramekin and place it on the plate",
    6: "pick up the black bowl on the cookie box and place it on the plate",
    7: "pick up the black bowl on the ramekin and place it on the plate",
    8: "pick up the black bowl on the stove and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}

LIBERO_OBJECT_TASKS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the bbq sauce and place it in the basket",
    2: "pick up the butter and place it in the basket",
    3: "pick up the chocolate pudding and place it in the basket",
    4: "pick up the cream cheese and place it in the basket",
    5: "pick up the ketchup and place it in the basket",
    6: "pick up the milk and place it in the basket",
    7: "pick up the orange juice and place it in the basket",
    8: "pick up the salad dressing and place it in the basket",
    9: "pick up the tomato sauce and place it in the basket",
}

# 反向字典：从任务描述到任务 ID
LIBERO_GOAL_TASK_TO_ID = {v: k for k, v in LIBERO_GOAL_TASKS.items()}
LIBERO_10_TASK_TO_ID = {v: k for k, v in LIBERO_10_TASKS.items()}
LIBERO_SPATIAL_TASK_TO_ID = {v: k for k, v in LIBERO_SPATIAL_TASKS.items()}
LIBERO_OBJECT_TASK_TO_ID = {v: k for k, v in LIBERO_OBJECT_TASKS.items()}


def instruction_to_task_id(instructions, task_suite_name="libero_goal"):
    """
    将 language instruction(s) 转换为 task_id
    
    Args:
        instructions: str or list/array of str - 任务的自然语言描述
    
    Returns:
        torch.Tensor: task_id(s) 范围 0-9
        如果是单个 instruction，返回 shape (1,) 的 tensor
        如果是 batch，返回 shape (batch_size,) 的 tensor
    
    Raises:
        ValueError: 如果 instruction 不在预定义的任务列表中
    """

    if task_suite_name == "libero_goal":
        task_to_id = LIBERO_GOAL_TASK_TO_ID
    elif task_suite_name == "libero_10":
        task_to_id = LIBERO_10_TASK_TO_ID
    elif task_suite_name == "libero_spatial":
        task_to_id = LIBERO_SPATIAL_TASK_TO_ID
    elif task_suite_name == "libero_object":
        task_to_id = LIBERO_OBJECT_TASK_TO_ID
    else:
        raise ValueError(f"Unknown task suite name: '{task_suite_name}'. Valid task suites: libero_goal, libero_object")

    # 处理单个 instruction
    if isinstance(instructions, str):
        if instructions not in task_to_id:
            raise ValueError(f"Unknown instruction: '{instructions}'. Valid instructions: {list(task_to_id.keys())}")
        return torch.tensor([task_to_id[instructions]], dtype=torch.long)
    
    # 处理 batch instructions
    task_ids = []
    for inst in instructions:
        # 处理可能的 numpy string 或 bytes
        if isinstance(inst, bytes):
            inst = inst.decode('utf-8')
        elif hasattr(inst, 'item'):  # numpy scalar
            inst = inst.item()
            if isinstance(inst, bytes):
                inst = inst.decode('utf-8')
        
        inst = str(inst).strip()
        
        if inst not in task_to_id:
            raise ValueError(f"Unknown instruction: '{inst}'. Valid instructions: {list(LIBERO_GOAL_TASK_TO_ID.keys())}")
        
        task_ids.append(task_to_id[inst])
    
    return torch.tensor(task_ids, dtype=torch.long)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def get_group_id(instruction):
    """根据单个指令判断所属的 Libero-10 分组 (1-6)"""
    if instruction is None: return None
    if hasattr(instruction, 'item'): instruction = instruction.item()
    if isinstance(instruction, bytes): instruction = instruction.decode('utf-8')
    ins = instruction.strip().lower()

    if "basket" in ins and any(x in ins for x in ["alphabet soup", "tomato sauce", "cream cheese", "butter"]): return 1
    if "stove" in ins and "moka pot" in ins: return 2
    if "black bowl" in ins and "bottom drawer" in ins: return 3
    if "white mug" in ins and ("plate" in ins or "chocolate pudding" in ins): return 4
    if "book" in ins and "caddy" in ins: return 5
    if "microwave" in ins: return 6
    return None

class RewModel_TaskEmbed_only4libero10(nn.Module):
    def __init__(self,
                 checkpoint_path: str = "/mnt/project_rlinf/jzn/workspace/latest/RLinf/reward_model/ckpt/libero_10_256_zsq/task_embed/seed42_dim64",
                 num_tasks: int = 10,
                 task_embed_dim: int = 64,
                 task_suite_name: str = "libero_10",
                 device: str = "cuda:0") -> None:
        super().__init__()
        print(f'init libero10 rwm - Preloading all models...')
        
        self.config = {
            "num_tasks": num_tasks,
            "task_embed_dim": task_embed_dim,
            "task_suite_name": task_suite_name
        }
        self.checkpoint_dir = checkpoint_path
        self.task_suite_name = task_suite_name
        self.device = torch.device(device)
        self.default_gid = "default"
        
        # 模型池
        self.models = nn.ModuleDict()
        
        # 你指定的 groups: 1,2,3,4,5,6
        self.possible_gids = ["1", "2", "3", "4", "5", "6"]
        
        # ==================== 关键改动：预加载所有模型 ====================
        for gid in self.possible_gids:
            gid_str = f"group_{gid}" if gid != "default" else "default"
            model = self._create_base_model().to(self.device)
            
            search_path = os.path.join(self.checkpoint_dir, gid_str) if gid != "default" else self.checkpoint_dir
            
            loaded = False
            if os.path.exists(search_path):
                pths = [f for f in os.listdir(search_path) if f.endswith('.pth')]
                if pths:
                    # 取最新的 checkpoint
                    pths.sort(key=lambda x: int(x[:-4]) if x[:-4].isdigit() else -1, reverse=True)
                    ckpt_path = os.path.join(search_path, pths[0])
                    
                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    model.load_state_dict(state_dict)
                    print(f"✓ Preloaded {gid_str} from {ckpt_path}")
                    loaded = True
            
            if not loaded:
                print(f"⚠ Warning: No checkpoint found for {gid_str}, using random weights")
            
            self.models[gid_str] = model
        
        print(f'init end - All {len(self.possible_gids)} models preloaded on {self.device}')

    def _create_base_model(self):
        """定义基础模型架构"""
        class SubModel(nn.Module):
            def __init__(self, num_tasks, task_embed_dim):
                super().__init__()
                b1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
                b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
                b3 = nn.Sequential(*resnet_block(64, 128, 2))
                b4 = nn.Sequential(*resnet_block(128, 256, 2))
                b5 = nn.Sequential(*resnet_block(256, 512, 2))
                
                self.visual_encoder = nn.Sequential(
                    b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                
                self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
                
                self.fusion_layer = nn.Sequential(
                    nn.Linear(512 + task_embed_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, obs, task_id):
                visual_features = self.visual_encoder(obs)
                task_embed = self.task_embedding(task_id)
                combined = torch.cat([visual_features, task_embed], dim=1)
                return self.fusion_layer(combined)

            def load_checkpoint(self, checkpoint_path):   # 保留以防后续需要
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.load_state_dict(checkpoint)
                print(f"SubModel loaded from {checkpoint_path}")

        return SubModel(self.config["num_tasks"], self.config["task_embed_dim"])

    def _get_model_by_gid(self, gid):
        """直接返回已预加载的模型"""
        gid_str = f"group_{gid}" if gid and gid != "default" else self.default_gid
        if gid_str not in self.models:
            raise ValueError(f"Group {gid_str} was not preloaded! Check possible_gids.")
        return self.models[gid_str]

    @torch.no_grad()
    def predict_rew(self, obs, instruction):
        obs = obs.clamp(-1.0, 1.0).to(dtype=torch.float32).to(self.device)
        batch_size = obs.shape[0]
        
        #print(f'instruction type: {type(instruction)}, value: {instruction[:3] if isinstance(instruction, (list, tuple)) else instruction}')  # 调试用
        
        # 统一处理指令为列表
        if isinstance(instruction, (str, bytes, np.str_)):
            instructions = [instruction] * batch_size
        else:
            instructions = list(instruction)   # 强制转 list
            
        # 计算 Group ID
        gids = [get_group_id(inst) for inst in instructions]
        unique_gids = set(gids)
        
        # 获取 Task IDs
        task_ids = instruction_to_task_id(instructions, self.task_suite_name).to(self.device)
        
        # 分组预测
        final_rewards = torch.zeros((batch_size, 1), device=self.device)
        
        for curr_gid in unique_gids:
            indices = [i for i, g in enumerate(gids) if g == curr_gid]
            if not indices:
                continue
                
            idx_tensor = torch.tensor(indices, device=self.device)
            sub_obs = obs[idx_tensor]
            sub_task_ids = task_ids[idx_tensor]
            
            model = self._get_model_by_gid(curr_gid)
            sub_rewards = model(sub_obs, sub_task_ids)
            
            final_rewards[idx_tensor] = sub_rewards
            
        return torch.round(final_rewards)

    def forward(self, obs=None, instruction=None):
        return self.predict_rew(obs, instruction)

class TaskEmbedResnetRewModel(nn.Module):
    def __init__(self,
                    checkpoint_path=None,
                    num_tasks: int = 10, 
                    task_embed_dim: int = 64, 
                    task_suite_name: str = "libero_goal") -> None:
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        self.visual_encoder = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten())  # 输出维度: (batch_size, 512)
        
        # 任务嵌入层 (0-9 的任务 ID)
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        
        # 融合层：将视觉特征和任务嵌入结合
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + task_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.task_suite_name = task_suite_name

        # 加载checkpoint（与RewModel保持一致）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    @torch.no_grad()
    def predict_rew(self, obs, instruction):
        """
        Args:
            obs: (batch_size, 3, height, width) 图像输入，范围 [-1, 1]
            instruction: str or list of str - 任务的自然语言描述
        
        Returns:
            reward: (batch_size, 1) 预测的奖励值，范围 [0, 1]
                   如果需要二值输出，在外部调用 torch.round()
        """
        # 使用 clamp 而不是 assert，避免运行时错误（与RewModel保持一致）
        obs = obs.clamp(-1.0, 1.0)
        
        # 转换instruction为task_id
        task_id = instruction_to_task_id(instruction, self.task_suite_name)
        if obs.device.type != 'cpu':
            task_id = task_id.to(obs.device)
        
        # 提取视觉特征
        visual_features = self.visual_encoder(obs.to(dtype=torch.float32))  # (batch_size, 512)

        # 获取任务嵌入
        task_embed = self.task_embedding(task_id)  # (batch_size, task_embed_dim)
        
        # 拼接视觉特征和任务嵌入
        combined_features = torch.cat([visual_features, task_embed], dim=1)  # (batch_size, 512 + task_embed_dim)
        
        # 预测奖励
        reward = self.fusion_layer(combined_features)
        
        # 可选：如果需要与RewModel完全一致，返回二值结果
        reward = torch.round(reward)
        
        return reward

    def forward(self, obs=None, instruction=None):
        """
        Forward pass for inference
        
        Args:
            obs: (batch_size, 3, height, width) 图像输入
            instruction: str or list of str - 任务的自然语言描述
        
        Returns:
            reward: 预测的奖励值
        """
        return self.predict_rew(obs, instruction)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")

class ResnetRewModel(nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
    ) -> None:
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            raise ValueError("checkpoint_path is required")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    @torch.no_grad()
    def predict_rew(self, obs):
        # assert obs.max() <= 1.5 and obs.min() >= -1.5, f"obs.max() is {obs.max()}, and obs min is {obs.min()}"
        obs = obs.clamp(-1.0, 1.0)
        x = self.net(obs.to(dtype=torch.float32))
        # 分为 0 或 1
        x = torch.round(x)
        return x

    def forward(self, obs=None):
        return self.predict_rew(obs)