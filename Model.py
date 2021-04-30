import parl
from parl import layers


# 策略网络
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act="relu")
        self.fc2 = layers.fc(size=act_dim, act="tanh")

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


# Q网络
class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act="relu")
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q


# 我们将策略网络和Q网络的类整合到一个无人机类里面去
class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # 获取策略网络的网络参数
    def get_actor_params(self):
        return self.actor_model.parameters()