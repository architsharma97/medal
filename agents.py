import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from networks import RandomShiftsAug, Encoder, DDPGActor, SACActor, Critic

class SACAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau,
                 reward_scale_factor, use_tb, from_vision):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.reward_scale_factor = reward_scale_factor
        self.use_tb = use_tb
        self.from_vision = from_vision
        # Changed log_std_bounds from [-10, 2] -> [-20, 2]
        self.log_std_bounds = [-20, 2]
        # Changed self.init_temperature to 1.0
        self.init_temperature = 1.0

        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = SACActor(model_repr_dim, action_shape, feature_dim,
                        hidden_dim, self.log_std_bounds).to(device)
        
        self.critic = Critic(model_repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(model_repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # Changed target entropy from -dim(A) -> -dim(A)/2
        self.target_entropy = -action_shape[0] / 2.0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
           # data augmentation
            self.aug = RandomShiftsAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, uniform_action=False, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        if self.from_vision:
            obs = self.encoder(obs.unsqueeze(0))

        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        
        if uniform_action:
            action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, discount, next_obs, step, not_done=None):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_V -= self.alpha.detach() * log_prob
            # TODO: figure out whether we want the not_done at the end or not
            target_Q = self.reward_scale_factor * reward + \
                            (discount * target_V * not_done.unsqueeze(1))


        Q1, Q2 = self.critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q + (self.alpha.detach() * log_prob)
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update_alpha(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss
            metrics['alpha_value'] = self.alpha

        return metrics

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, step_type, next_step_type = utils.to_torch(batch, self.device)

        return (obs, action, reward, discount, next_obs, step_type, next_step_type)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple

        not_done = next_step_type.clone()
        not_done[not_done < 2] = 1
        not_done[not_done == 2] = 0

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

class DDPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 stddev_schedule, stddev_clip, use_tb, from_vision):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.from_vision = from_vision
        self.log_std_bounds = [-10, 2]

        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = DDPGActor(model_repr_dim, action_shape, feature_dim,
                               hidden_dim, self.log_std_bounds).to(device)
        
        self.critic = Critic(model_repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(model_repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
           # data augmentation
            self.aug = RandomShiftsAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
 
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, uniform_action=False, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        if self.from_vision:
            obs = self.encoder(obs.unsqueeze(0))

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        
        if uniform_action:
            action.uniform_(-1.0, 1.0)
            
        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + discount * target_V


        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
    
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, step_type, next_step_type = utils.to_torch(batch, self.device)

        return (obs, action, reward, discount, next_obs, step_type, next_step_type)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

class MEDALBackwardAgent(SACAgent):
    def __init__(self, *agent_args, discrim_hidden_size=128, discrim_lr=3e-4, mixup=True, discrim_eps=1e-10, **agent_kwargs):
        
        super(MEDALBackwardAgent, self).__init__(**agent_kwargs)
        self.discrim_hidden_size = discrim_hidden_size
        self.discrim_lr = discrim_lr
        self.discrim_eps  = discrim_eps
        self.mixup = mixup
        self.discriminator = nn.Sequential(nn.Linear(self.obs_shape[0], discrim_hidden_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(discrim_hidden_size, 1)).to(self.device)

        self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)

    def update_discriminator(self, pos_replay_iter, neg_replay_iter):
        if self.from_vision:
            print("update_discrim does not support vision")
            exit()

        metrics = dict()
        
        batch_pos = next(pos_replay_iter)
        obs_pos, _, _, _, _, _, _ = utils.to_torch(batch_pos, self.device)
        num_pos = obs_pos.shape[0]

        batch_neg = next(neg_replay_iter)
        obs_neg, _, _, _, _, _, _ = utils.to_torch(batch_neg, self.device)
        num_neg = obs_neg.shape[0]

        if self.mixup:
            alpha = 1.0
            beta_dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
            
            l = beta_dist.sample([num_pos + num_neg])
            mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
            
            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)
            disc_inputs = torch.cat((obs_pos, obs_neg), 0)

            # TODO: is this the fastest way to do things?
            ridxs = torch.randperm(num_pos + num_neg)
            perm_labels = labels[ridxs]
            perm_disc_inputs = disc_inputs[ridxs]

            images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
            labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
        else:
            images = torch.cat((obs_pos, obs_neg), 0)
            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

        loss = torch.nn.BCELoss()
        m = nn.Sigmoid()
        discrim_loss = loss(m(self.discriminator(images)), labels)

        self.discrim_opt.zero_grad(set_to_none=True)
        discrim_loss.backward()
        self.discrim_opt.step()
        
        if self.use_tb:
            metrics['discriminator_loss'] = discrim_loss.item()

        return metrics

    def compute_reward(self, obs):
        actual_reward = -torch.log(1 - torch.sigmoid(self.discriminator(obs)) + self.discrim_eps)
        return actual_reward

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, step_type, next_step_type = utils.to_torch(batch, self.device)

        return (obs, action, self.compute_reward(next_obs).detach(), discount, next_obs, step_type, next_step_type)
