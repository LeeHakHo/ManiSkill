import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALGO_NAME = "BC_Diffusion_rgbd_Dita_DDP"

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import worker_init_fn
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv

from llama_dp import RobotTransformerNet
from transformers import AutoTokenizer, CLIPModel
import timm


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "StackCube-v1"
    demo_path: str = "demos/StackCube-v1/trajectory.h5" 
    num_demos: Optional[int] = None
    total_iters: int = 30000 

    batch_size: int = 32  

    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    
    obs_mode: str = "rgb+depth" 
    max_episode_steps: Optional[int] = None
    log_freq: int = 100
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 50
    num_eval_envs: int = 10

    sim_backend: str = "gpu"
    num_dataload_workers: int = 4

    control_mode: str = "pd_joint_pos"
    
    instruction_text: str = "Stack the red cube on the blue cube"
    clip_text_model_name: str = "openai/clip-vit-base-patch32"
    freeze_clip: bool = True


# =========================
# Wrapper
# =========================
class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]

        if "agent" in observation:
            for key in ["world__T__ee", "world__T__root"]:
                if key in observation["agent"]:
                    del observation["agent"][key]

        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(cam_data["rgb"].permute(0, 3, 1, 2))
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)

        ret = {}
        flat_state = common.flatten_state_dict(observation, use_torch=True)

        if self.include_state:
            ret["state"] = flat_state

        if self.include_rgb:
            ret["rgb"] = torch.stack(images_rgb, dim=1) 

        if self.include_depth:
            ret["depth"] = torch.stack(images_depth, dim=1) 

        if "agent_pos" not in ret:
            ret["agent_pos"] = ret.get("state", flat_state)

        return ret


# =========================
# Dataset
# =========================
class SmallDemoDataset_Dita_ACTStyle(Dataset):
    def __init__(self, data_path, device, num_traj, include_rgb, include_depth, obs_horizon, pred_horizon, control_mode):
        self.data_path = data_path
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        from diffusion_policy.utils import load_demo_dataset
        print(f"Loading dataset from {data_path} into RAM...")
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)

        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        obs_traj_dict_list = []
        for obs_traj in tqdm(trajectories["observations"], desc="Pre-processing obs"):
            processed_obs = self.process_obs(obs_traj)
            obs_traj_dict_list.append(processed_obs)
        trajectories["observations"] = obs_traj_dict_list

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])

        self.slices = []
        for traj_idx in range(len(trajectories["actions"])):
            L = trajectories["actions"][traj_idx].shape[0]
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

            if traj_idx == 0:
                action_dim = trajectories["actions"][0].shape[1]
                if "delta_pos" in control_mode or "pd_joint_vel" in control_mode:
                    self.pad_action_arm = torch.zeros((action_dim - 1,))
                elif "pd_joint_pos" in control_mode:
                    self.pad_action_arm = None
                else:
                    pass

        self.trajectories = trajectories
        print(f"Dataset Loaded. Total samples: {len(self.slices)}")

    def process_obs(self, obs_dict):
        sensor_data = obs_dict.pop("sensor_data")
        images_rgb = []
        images_depth = []

        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb = torch.from_numpy(cam_data["rgb"])
                resized_rgb = self.transforms(rgb.permute(0, 3, 1, 2))
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.Tensor(cam_data["depth"].astype(np.float32) / 1024)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)

        ret = {}
        if self.include_rgb:
            ret["rgb"] = torch.stack(images_rgb, dim=1)
        if self.include_depth:
            ret["depth"] = torch.stack(images_depth, dim=1)

        obs_dict["extra"] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict["extra"].items()}
        state = common.flatten_state_dict(obs_dict, use_torch=True)
        ret["state"] = state
        return ret

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        L = self.trajectories["actions"][traj_idx].shape[0]

        obs_seq = {}
        for k, v in obs_traj.items():
            if start < 0:
                pad_len = abs(start)
                actual_data = v[0 : start + self.obs_horizon]
                pad_data = v[0:1].repeat(pad_len, *([1] * (v.ndim - 1)))
                obs_seq[k] = torch.cat([pad_data, actual_data], dim=0)
            else:
                obs_seq[k] = v[start : start + self.obs_horizon]

        act_read_start = max(0, start)
        act_read_end = min(L, end)
        actions = self.trajectories["actions"][traj_idx][act_read_start:act_read_end]

        pad_len_act = act_read_start - start
        if pad_len_act > 0:
            pad_vec = actions[0:1].repeat(pad_len_act, 1)
            actions = torch.cat([pad_vec, actions], dim=0)

        if end > L:
            pad_len_end = end - L
            if hasattr(self, 'pad_action_arm') and self.pad_action_arm is not None:
                gripper_action = actions[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                pad_vec = pad_action.unsqueeze(0).repeat(pad_len_end, 1)
            else:
                pad_vec = actions[-1:].repeat(pad_len_end, 1)
            actions = torch.cat([actions, pad_vec], dim=0)

        return {"observations": obs_seq, "actions": actions}

    def __len__(self):
        return len(self.slices)


class DitaAgent(nn.Module):
    def __init__(self, env, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon

        if env is not None:
            action_space = env.single_action_space
        else:
            tmp_env = gym.make(args.env_id, obs_mode=args.obs_mode)
            tmp_env = FlattenRGBDObservationWrapper(tmp_env, rgb=True, depth=("depth" in args.obs_mode), state=True)
            action_space = tmp_env.action_space
            tmp_env.close()

        self.act_dim = action_space.shape[0]
        act_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        act_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)
        self.register_buffer("act_mid", (act_low + act_high) / 2.0)
        self.register_buffer("act_scale", (act_high - act_low) / 2.0 + 1e-6)

        self.time_sequence_length = self.obs_horizon + self.pred_horizon - 1

        self.register_buffer("clip_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        clip_model = CLIPModel.from_pretrained(args.clip_text_model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.clip_text_model_name)
        clip_text_encoder = clip_model.text_model

        with torch.no_grad():
            inputs = tokenizer(
                text=args.instruction_text,
                return_tensors="pt",
                max_length=77,
                padding="max_length",
                truncation=True,
            )
            text_emb = clip_text_encoder(**inputs)[0].squeeze(0)

        self.register_buffer("instruction_tokens", text_emb, persistent=True)
        del clip_model, clip_text_encoder, tokenizer

        vision_backbone_dim = 768
        self.text_projector = nn.Linear(512, vision_backbone_dim)

        self.net = RobotTransformerNet(
            output_tensor_spec=None,
            vocab_size=self.instruction_tokens.shape[1],
            trajectory_dim=self.act_dim,
            token_embedding_size=vision_backbone_dim,
            intermediate_size=vision_backbone_dim * 4,
            hidden_size=vision_backbone_dim,
            num_layers=8,
            dropout_rate=0.1,
            time_sequence_length=self.time_sequence_length,
            input_size="(224, 224)",
            include_prev_timesteps_actions=False,
            freeze_backbone=True,
            use_qformer=True, 
            use_wrist_img=False,
            use_depth_img=False,
            dim_align_type=0,
            prediction_type="epsilon",
            scheduler_type=1,
            num_inference_steps=10, 
            attn_implementation="eager",
            use_action_head_diff=2,
        )
        
        self.noise_scheduler = self.net.noise_scheduler
        self.noise_scheduler_eval = self.net.noise_scheduler_eval

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp((a - self.act_mid) / self.act_scale, -1.0, 1.0)

    def _denormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.act_scale + self.act_mid

    def _ensure_btc_hw(self, x: torch.Tensor, expect_c: int = 3) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[-1] == expect_c and x.shape[1] != expect_c:
                x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(1)
        elif x.ndim == 5:
            x = x[:, 0]
            x = x.unsqueeze(1)
        elif x.ndim == 6:
            x = x[:, :, 0]
        else:
            raise RuntimeError(f"Unsupported dim: {x.ndim}")
        return x

    def _build_obs_for_dita(self, obs_seq):
        dev = next(self.parameters()).device

        rgb = obs_seq["rgb"]
        if not torch.is_tensor(rgb):
            rgb = torch.as_tensor(rgb)

        rgb = self._ensure_btc_hw(rgb, expect_c=3).to(dev).float()

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        rgb = (rgb - self.clip_mean) / self.clip_std
        obs = {"image": rgb}

        if "depth" in obs_seq:
            depth = obs_seq["depth"]
            if not torch.is_tensor(depth):
                depth = torch.as_tensor(depth)
            depth = self._ensure_btc_hw(depth, expect_c=1).to(dev).float()
            if depth.shape[2] == 1:
                depth = depth.repeat(1, 1, 3, 1, 1)
            obs["depth_image"] = depth
            obs["rgbd_image"] = torch.cat([rgb, depth[:, :, :3]], dim=2)

        L, D = self.instruction_tokens.shape
        lang_tokens = self.instruction_tokens.to(dev).unsqueeze(0).unsqueeze(0).expand(rgb.shape[0], rgb.shape[1], L, D)
        lang_tokens = self.text_projector(lang_tokens)
        obs["natural_language_embedding"] = lang_tokens

        if "state" in obs_seq:
            s = obs_seq["state"]
            if not torch.is_tensor(s):
                s = torch.as_tensor(s)
            obs["agent_pos"] = s.to(dev).float()

        return obs

    def forward(self, obs_seq, action_seq):
        dev = next(self.parameters()).device
        traj = self._normalize_action(action_seq)
        B = traj.shape[0]
        noise = torch.randn_like(traj, device=dev)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=dev,
        ).long()

        noisy_traj = self.noise_scheduler.add_noise(traj, noise, timesteps)
        obs = self._build_obs_for_dita(obs_seq)

        pred = self.net(
            obs,
            act=None,
            noisy_action_tokens=noisy_traj,
            timesteps=timesteps,
            num_pred_action=self.pred_horizon,
        )
        loss = F.mse_loss(pred, noise)
        return loss

    @torch.no_grad()
    def get_action(self, obs_seq):
        self.eval()
        dev = next(self.parameters()).device

        obs = self._build_obs_for_dita(obs_seq)
        image = obs["image"]
        B = image.shape[0]

        traj = torch.randn(
            size=(B, self.pred_horizon, self.act_dim),
            dtype=image.dtype,
            device=dev,
        )

        num_inference_steps = self.net.num_inference_steps
        self.noise_scheduler_eval.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler_eval.timesteps:
            t_batch = t.repeat(B).to(dev)
            model_output = self.net(
                obs,
                act=None,
                noisy_action_tokens=traj,
                timesteps=t_batch,
                num_pred_action=self.pred_horizon,
                ret_feats=False,
            )
            traj = self.noise_scheduler_eval.step(
                model_output,
                t,
                traj,
                generator=None,
            ).prev_sample

        traj_denorm = self._denormalize_action(traj)
        start = self.pred_horizon - self.act_horizon
        end = self.pred_horizon
        actions = traj_denorm[:, start:end, :]
        return actions


def save_ckpt(run_name, tag, agent, ema_agent, rank):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    if rank == 0:
        # Save both standard model and EMA model
        torch.save({
            "agent": agent.module.state_dict() if isinstance(agent, DDP) else agent.state_dict(),
            "ema_agent": ema_agent.state_dict()
        }, f"runs/{run_name}/checkpoints/{tag}.pt")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank()

    args = tyro.cli(Args)

    if rank == 0:
        print(f"Running in DDP Mode. World Size: {dist.get_world_size()}")

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = None
    if rank == 0:
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )
        if args.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = args.max_episode_steps

        other_kwargs = dict(obs_horizon=args.obs_horizon)

        wrappers = [partial(FlattenRGBDObservationWrapper, rgb=True, depth=("depth" in args.obs_mode), state=True)]
        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=wrappers,
        )

    if args.track and rank == 0:
        import wandb
        config = vars(args)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="Dita",
            tags=["dita", "diffusion_policy", "ddp", "ema"],
        )

    writer = SummaryWriter(f"runs/{run_name}") if rank == 0 else None

    include_rgb = True
    include_depth = ("depth" in args.obs_mode)

    dataset = SmallDemoDataset_Dita_ACTStyle(
        args.demo_path,
        device,
        args.num_demos,
        include_rgb,
        include_depth,
        args.obs_horizon,
        args.pred_horizon,
        args.control_mode,
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_dataload_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
        drop_last=True,
    )

    # Agent setup
    agent = DitaAgent(None, args).to(device)

    # ✅ EMA 설정 (DDP 이전/이후 어디든 상관없으나 보통 모델 wrapping 전후로 함)
    # diffusers의 EMAModel은 파라미터를 복사해서 유지함
    # H100 80GB가 충분하므로 각 GPU마다 EMA 복사본을 가져도 문제 없음
    ema = EMAModel(
        parameters=agent.parameters(),
        power=0.75, # ACT 코드에 맞춰 0.75로 설정 (Dita 기본값 확인 필요, 보통 0.9999 많이 씀)
        model_cls=DitaAgent,
        model_config=None 
    )
    ema_agent = DitaAgent(None, args).to(device) # 평가용 껍데기 모델
    ema_agent.eval()

    # ✅ DDP Wrap
    # Vision Backbone이 Freeze 되어 있으므로 find_unused_parameters=False가 효율적일 수 있으나,
    # 일부 모듈이 안 쓰일 경우 안전을 위해 True로 두기도 함. 여기선 Backbone이 freeze라 False 추천.
    agent = DDP(agent, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    iterator = iter(train_dataloader)
    current_iter = 0
    pbar = tqdm(total=args.total_iters) if rank == 0 else None

    while current_iter < args.total_iters:
        try:
            data_batch = next(iterator)
        except StopIteration:
            sampler.set_epoch(current_iter // max(1, len(train_dataloader)))
            iterator = iter(train_dataloader)
            data_batch = next(iterator)

        obs_gpu = {}
        for k, v in data_batch["observations"].items():
            if isinstance(v, torch.Tensor):
                obs_gpu[k] = v.to(device, non_blocking=True)
        act_gpu = data_batch["actions"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            total_loss = agent(obs_gpu, act_gpu)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        # ✅ Update EMA
        ema.step(agent.parameters())

        if rank == 0 and (current_iter % args.log_freq == 0):
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], current_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), current_iter)

        if current_iter > 0 and (current_iter % args.eval_freq == 0):
            dist.barrier()
            # ✅ EMA 가중치로 평가 수행
            if rank == 0:
                print("Copying EMA weights for evaluation...")
                ema.copy_to(ema_agent.parameters())
                
                print("Running evaluation on GPU...")
                eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)

                print(f"Iter {current_iter}: Success Rate {np.mean(eval_metrics['success_at_end']):.4f}")
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"eval/{k}", np.mean(v), current_iter)

                # 체크포인트 저장 (EMA 포함)
                save_ckpt(run_name, str(current_iter), agent, ema_agent, rank)

            dist.barrier()

        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({"loss": total_loss.item()})

        current_iter += 1

    if rank == 0:
        envs.close()
        writer.close()

    dist.destroy_process_group()