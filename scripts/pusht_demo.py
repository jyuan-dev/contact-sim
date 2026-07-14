import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human", max_episode_steps=5000, damping=0.3)
observation, info = env.reset()

env.unwrapped.screen = env.unwrapped.window  # needed for teleop_agent's coordinate conversion
teleop = env.unwrapped.teleop_agent()

env.unwrapped.k_p = 100  # default value
env.unwrapped.k_v = 20   # default value

for _ in range(10000):
    act = teleop.act(observation)
    if act is None:
        act = env.unwrapped.agent.position  # stay in place until mouse is close enough
    observation, reward, terminated, truncated, info = env.step(act)
    env.render()

    if terminated or truncated:
        print(f"terminated={terminated}, truncated={truncated}, coverage={info.get('coverage')}, step={_}")
        observation, info = env.reset()

env.close()