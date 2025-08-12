import os

os.environ["MUJOCO_GL"] = "osmesa"

# roboosuite 환경 구동 테스트
import robosuite as suite
import numpy as np


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=False
)

obs = env.reset()

low, high = env.action_spec

for i in range(10):
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1} | Reward: {reward:.3f} | Done: {done}")

env.close()

# LD_LIBRARY_PATH 설정 충돌 테스트

# import torch

# def test_cuda():
#     print("PyTorch Version:", torch.__version__)
#     print("CUDA Available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         x = torch.rand(3, 3).cuda()
#         y = torch.mm(x, x)
#         print("Matrix product succeeded. Device:", x.device)
#     else:
#         print("CUDA is NOT available.")

# if __name__ == "__main__":
#     test_cuda()
