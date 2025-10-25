from gem.envs.registration import make

# Initialize the environment
env = make("game:GuessTheNumber-v0-easy")
from logging import getLogger

logger = getLogger(__name__)

# Reset the environment to generate the first observation
observation, info = env.reset()
for episode in range(30):
    action = env.sample_random_action() # insert policy here

    # apply action and receive next observation, reward
    # and whether the episode has ended
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()


    logger.info(f"""
                Step: {episode}, 
                  Action: {action}, 
                  Observation: {observation},
                  Info: {info},
                  Reward: {reward},
                  Terminated: {terminated}
                  Truncated: {truncated}
""")