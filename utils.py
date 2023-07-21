import gymnasium as gym


def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk
