""" This script is used to run a ACT policy on the real robot."""

import rospy
import torch
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

class ACTRealInference:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = torch.device(device)
        
        # 모델 체크포인트 로드
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.stats = {k: v.to(self.device) for k, v in ckpt['norm_stats'].items()}
        
        # Agent 초기화 (envs와 args는 학습 시 설정값과 동일해야 함)
        self.agent = Agent(envs, args).to(self.device)
        self.agent.load_state_dict(ckpt['ema_agent'])
        self.agent.eval()

        # Temporal Aggregation 설정
        self.num_queries = 30 # args.num_queries
        self.all_time_actions = torch.zeros([self.num_queries, self.num_queries, 8]).to(self.device)
        self.step_idx = 0

    def get_real_action(self, rgb_img, robot_state):
        # A. 전처리 (Resize 224, Normalize)
        # node.preprocess_ros_image(rgb_img) 등의 로직 적용
        obs = {
            'rgb': self.preprocess(rgb_img),
            'state': self.process_state(robot_state) # 18차원 벡터
        }

        # B. 추론
        with torch.no_grad():
            action_chunk = self.agent.get_action(obs) # (1, 30, 8)

        # C. Temporal Aggregation (시간차 가중 평균)
        self.all_time_actions[self.step_idx % self.num_queries] = action_chunk[0]
        
        actions_for_curr_step = []
        for i in range(self.num_queries):
            actions_for_curr_step.append(self.all_time_actions[i, (self.step_idx - i) % self.num_queries])
        
        weights = torch.exp(-0.01 * torch.arange(self.num_queries)).to(self.device)
        weights = weights / weights.sum()
        raw_action = (torch.stack(actions_for_curr_step) * weights.unsqueeze(-1)).sum(dim=0)

        # D. 역정규화
        action = (raw_action * self.stats['action_std']) + self.stats['action_mean']
        self.step_idx += 1
        return action.cpu().numpy()


if __name__ == "__main__":

    rospy.init_node("realsense_subscriber", anoymous=False)
    logger = DemonstrationLogger(args.camer_namespace)
    logger.run()


    ckpt_path = "code/ManiSkill/real_model/.../75000.pt"
    node = ManiSkillRealRobotNode(ckpt_path)




    while True:
        rgb = logger.latest_rgb()
        traj = inference_node.get_real_action(rgb, state)
        
        
        if rgb is None or state is None:
            continue
        
        
        for i in range(action_horizon):
            action = traj[i]
            #send to controller
            end_to_franka(action)