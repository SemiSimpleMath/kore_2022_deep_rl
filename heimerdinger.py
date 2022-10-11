import collections
import contextlib
import copy
import datetime
import gzip
import inspect
import logging
import os
import random
import shutil
import sys
import tarfile
from typing import Any, Dict, Optional, Union

import kaggle_environments.envs.kore_fleets.helpers as kf
import numpy as np
import pytz
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as tdist

from board_symmetries import apply_random_symmetry
from encode_flightplan import PlanEncoder, BoardWrapper
import utils


if '__file__' in globals():
    logger = logging.getLogger(__file__)
else:
    logger = logging.getLogger('heimerdinger.py')
logging.basicConfig(
    level=logging.INFO,
    format='{asctime} {name} {levelname:8s} {message}',
    style='{',
    handlers=[
        # logging.FileHandler("debug.log"),
        # logging.StreamHandler(sys.stdout)
    ]
)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.ERROR)
logger.addHandler(sh)

if os.path.exists("/tmp"):
    fh = logging.FileHandler("/tmp/heimerdinger.log", mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


perf = utils.PerfLogger(logger)


class ResConvBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.model = nn.Sequential(
            utils.ResidualBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(num_features=width),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(num_features=width),
                ),
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class BoardRepresentation(nn.Module):

    def __init__(self, width=256, num_res_blocks=21, skip_np_encode=False):
        super().__init__()
        self.latent_size = utils.ENC_PLAN_DIMS
        self.skip_np_encode = skip_np_encode
        self.plan_encoder = BoardWrapper(
            PlanEncoder(
                raw_size=utils.ENC_PLAN_DIMS,
                latent_size=self.latent_size
            )
        )
        # in: (12 + latent_size) x H x W
        # out: 256 x H x W
        if skip_np_encode:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=12 + self.latent_size,
                    out_channels=width-12, kernel_size=3, padding=1, padding_mode='circular'
                ),
                nn.BatchNorm2d(num_features=width-12),
                nn.ReLU(),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=12 + self.latent_size,
                    out_channels=width, kernel_size=3, padding=1, padding_mode='circular'
                ),
                nn.BatchNorm2d(num_features=width),
                nn.ReLU(),
            )

        self.model = nn.Sequential(conv_block)

        # in: 256 x H x W
        # out: 256 x H x W
        # in: 256 x H x W
        # out: 256 x H x W
        for i in range(num_res_blocks):
            if skip_np_encode:
                self.model.add_module(f'ResidualBlock_{i}', ResConvBlock(width-12))
            else:
                self.model.add_module(f'ResidualBlock_{i}', ResConvBlock(width))

    def forward(self, nonplans, plans):
        # nonplans: 10 x H x W
        # plans: H x W x raw_size
        enc_plans = self.plan_encoder(plans)
        enc_plans = enc_plans.swapaxes(-1, -2).swapaxes(-2, -3)

        # (10 + latent_size) x H x W
        combined = torch.cat([nonplans, enc_plans], dim=-3)

        # BatchNorm2D only takes 4d tensors...
        ret = self.model(combined.unsqueeze(0))
        if not ret.isfinite().all():
            logger.critical(f'Got non-finite in BoardRepresentation')
        if self.skip_np_encode:
            return torch.cat([nonplans, ret.squeeze(0)], dim=-3).unsqueeze(0)
        else:
            return ret


class Actor(nn.Module):

    def __init__(self, width=256):
        super().__init__()
        # in: 256 x 21 x 21
        # out: (1+10+1) (1-D array)
        # noinspection PyTypeChecker
        self.model = nn.Sequential(*([
            ResConvBlock(width)
            for _ in range(21)
        ] + [
            nn.Conv2d(in_channels=width, out_channels=utils.NUM_PLAN_TYPES + 23, kernel_size=1),
            nn.BatchNorm2d(utils.NUM_PLAN_TYPES + 23),
        ]))

    def raw_forward(self, rep):
        return self.model(rep)

    def forward(self, rep):
        # rep: 256 x 21 x 21
        # out: NUM_PLAN_TYPES+23 x 21 x 21
        raw = self.model(rep)
        raw[:, :12, :, :] = F.softmax(raw[:, :12, :, :], dim=1)
        raw[:, 12:15, :, :] = torch.sigmoid(raw[:, 12:15, :, :])
        raw[:, 15:15 + utils.NUM_PLAN_TYPES, :, :] = F.softmax(raw[:, 15:15 + utils.NUM_PLAN_TYPES, :, :], dim=1)
        raw[:, 15 + utils.NUM_PLAN_TYPES:, :, :] = F.softmax(raw[:, 15 + utils.NUM_PLAN_TYPES:, :, :], dim=1)
        return raw


class Critic(nn.Module):

    def __init__(self, width=256):
        super().__init__()
        # noinspection PyTypeChecker
        self.model = nn.Sequential(*([
            ResConvBlock(width)
            for _ in range(21)
        ] + [
            nn.Conv2d(in_channels=width, out_channels=64, kernel_size=5),  # 17
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5),  # 13
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5),  # 9
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5),  # 5
            nn.AvgPool2d(kernel_size=(5, 5)),
            nn.Flatten(start_dim=-3),
            nn.Tanh()
        ]))

    def forward(self, rep, actions):
        combined = torch.cat([rep, actions], dim=-3)
        return self.model(combined)


class HeimerdingerAgent(nn.Module):

    def __init__(self, width=64, num_res_blocks=21, ddpg=False):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        self.tail = BoardRepresentation(width=width, num_res_blocks=num_res_blocks, skip_np_encode=False)

        # in: N x 269 x H x W
        # out: N x (12 + NUM_PLAN_TYPES + 11)
        self.actor = Actor(width=width)
        self.actor_target = None

        # in: N x 256 x H x W
        # out: N x 1
        self.critic = Critic(width=width + utils.NUM_PLAN_TYPES + 23)
        self.critic_target = None
        self.ddpg = ddpg
        if self.ddpg:
            self.use_ddpg()
        self.board: Optional[kf.Board] = None
        self.kore_spent_this_turn = 0.
        self.ships_spawned_this_turn = 0
        self.ships_launched_this_turn = 0

        self.last_pol: Optional[torch.Tensor] = None
        self.apply_symmetries = False
        self.save_dir = None
        self.log_counts = collections.defaultdict(int)
        self.log_stats_every = 79  # turns
        self.checkpoint_tag = None  # tagname of the model that's currently loaded
        self.epsilon = 0.05
        self.explore = False

        self.last_critic_loss = 0
        self.last_actor_loss = 0
        self.num_train_cycles = 0
        self.log_every = 10
        self.gamma = 0.99            # discount factor
        self.tau = 1e-3              # for soft update of target parameters

        # currently unused
        self.explore_sigma_min = .05
        self.explore_sigma_max = 1
        self.num_explore_steps = 10
        self.name = None  # this is used in self play

        self.log_losses_every = 10
        self._cuda = False
        if torch.cuda.is_available():
            self.to_cuda()

        self.tail_opt = torch.optim.Adam(
            self.tail.parameters(),
            lr=1.5e-4,
            weight_decay=1e-4,
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=3e-4,
            weight_decay=1e-3,
        )

        if os.path.exists('dinger.pt'):
            self.load_state_dict(torch.load('dinger.pt'))

    @property
    def encoded_board(self) -> (torch.Tensor, torch.Tensor):
        nonplans, plans = utils.encode_board(self.board)
        if self.is_cuda:
            nonplans = nonplans.cuda()
            plans = plans.cuda()
        return nonplans, plans

    @property
    def me(self):
        return self.board.current_player

    @property
    def kore(self):
        return self.me.kore

    def to_cuda(self):
        self._cuda = True
        utils.to_cuda(self)

    def to_cpu(self):
        self._cuda = False
        utils.to_cpu(self)

    def use_ddpg(self):
        self.ddpg = True
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

    @property
    def is_cuda(self):
        return self._cuda

    @contextlib.contextmanager
    def train_ctx(self):
        """Context manager for putting all the networks in train mode. The way to use this is

            with agent.train_ctx():
                agent.postgame_update(...)

        See usage in train.py.
        """
        training = self.training
        try:
            self.train()
            yield
        finally:
            self.train(training)

    @contextlib.contextmanager
    def eval_ctx(self):
        """Context manager for putting all the networks in eval mode. The way to use this is

            with agent.eval_ctx():
                agent.postgame_update(...)
        """
        training = self.training
        try:
            with torch.no_grad():
                self.eval()
                yield
        finally:
            self.train(training)

    def save(self, filename=None):
        """Saves agent model weights to specified directory, and then puts them in a tarball and zips them up."""
        assert self.save_dir is not None, "Need to set self.save_dir before saving."
        ct = pytz.timezone('America/Chicago')
        now = ct.localize(datetime.datetime.utcnow())
        nowstr = now.strftime('%Y%m%d-%H%M')
        # we could gzip this, but tarring them takes 150 ms and gzipping them takes 4.5 seconds
        if filename is None:
            tarname = self.save_dir / f'{type(self).__name__}-{nowstr}.tar'.lower()
        else:
            tarname = self.save_dir / f'{type(self).__name__}-{filename}.tar'.lower()
        with tarfile.open(tarname, 'w') as f:
            for k, v in vars(self).items():
                if isinstance(v, nn.Module):
                    fn = self.save_dir / f'{k}-{nowstr}.pt'
                    torch.save(v.state_dict(), fn)
                    f.add(fn)
                    os.remove(fn)
        self.checkpoint_tag = nowstr
        return tarname

    def load(self, path: str):
        """Loads either gzipped (.tar.gz) or tarball (.tar) weights file into the model."""
        if path.endswith('.gz'):
            unzipped_path = path[:-3]  # heimerdinger-20220612-0605.tar.gz -> heimerdinger-20220612-0605.tar
            if not os.path.exists(unzipped_path):
                with gzip.open(path, 'rb') as g:
                    with open(unzipped_path, 'wb') as f:
                        shutil.copyfileobj(g, f)
        else:
            unzipped_path = path
        assert unzipped_path.endswith('.tar'), repr(unzipped_path)
        with tarfile.open(unzipped_path) as f:
            names = f.getnames()
            f.extractall()
        for name in names:
            compname = name.split('-')[0]
            comp = getattr(self, compname, None)
            if isinstance(comp, nn.Module):
                comp.load_state_dict(torch.load(name))
            os.remove(name)
        self.checkpoint_tag = unzipped_path[:-4].split('-')[-1]

    def log_critical(self, msg, every=1):
        lineno = inspect.currentframe().f_back.f_lineno
        if self.log_counts[lineno] % every == 0:
            logger.critical(msg)
        self.log_counts[lineno] += 1

    def log_error(self, msg, every=1):
        lineno = inspect.currentframe().f_back.f_lineno
        if self.log_counts[lineno] % every == 0:
            logger.error(msg)
        self.log_counts[lineno] += 1

    def log_warning(self, msg, every=1):
        lineno = inspect.currentframe().f_back.f_lineno
        if self.log_counts[lineno] % every == 0:
            logger.warning(msg)
        self.log_counts[lineno] += 1

    def log_info(self, msg, every=1):
        lineno = inspect.currentframe().f_back.f_lineno
        if self.log_counts[lineno] % every == 0:
            logger.info(msg)
        self.log_counts[lineno] += 1

    def log_debug(self, msg, every=1):
        lineno = inspect.currentframe().f_back.f_lineno
        if self.log_counts[lineno] % every == 0:
            print(msg)
        self.log_counts[lineno] += 1

    def display_param_count(self):
        print('total parameters (heimerdinger):')
        print(f'   (all):        {sum(p.numel() for p in self.parameters()):11,.0f}')
        print(f'      (tail):    {sum(p.numel() for p in self.tail.parameters()):11,.0f}')
        print(f'      (actor):   {sum(p.numel() for p in self.actor.parameters()):11,.0f}')
        print(f'      (critic):  {sum(p.numel() for p in self.critic.parameters()):11,.0f}')

    def encode_board(self, board: kf.Board, actions: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        np_t, p_t = utils.encode_board(board)
        if self.is_cuda:
            np_t = np_t.cuda()
            p_t = p_t.cuda()
        if not self.apply_symmetries:
            return np_t, p_t, actions
        transform_params, nonplan_data, plan_data, action_data = apply_random_symmetry(np_t, p_t, actions.unsqueeze(0))
        return transform_params, nonplan_data, plan_data, action_data.squeeze(0)

    @staticmethod
    def sample_buffer(buffer, batch_size=128):
        """Randomly sample a batch of experiences from memory."""
        batch_keys = random.sample(list(buffer), k=batch_size)
        obs = []
        proto_actions = []
        rewards = []
        new_obs = []
        dones = []
        for step in batch_keys:
            obs.append(buffer[step]["obs"])
            proto_actions.append(buffer[step]["proto_action"])
            new_obs.append(buffer[step]["new_obs"])
            rewards.append((torch.ones(1) * buffer[step]["reward"]))
            dones.append((torch.ones(1) * buffer[step]["done"]))

        proto_actions = torch.stack(proto_actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        dones = torch.stack(dones, dim=0)

        return obs, proto_actions, rewards, new_obs, dones

    def ddpg_update(self, replay_buffer, batch_size=128):
        assert self.ddpg, "DDPG not active."

        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                gamma (float): discount factor
        """

        obs, proto_actions, rewards, new_obs, dones = self.sample_buffer(replay_buffer, batch_size)
        states = []
        next_states = []
        actions_next = []
        for o, o_new in zip(obs, new_obs):
            board = kf.Board(o, self.board.configuration)
            _, np_t, p_t, a = self.encode_board(board, None)
            states.append(self.tail.forward(np_t, p_t))
            new_board = kf.Board(o_new, self.board.configuration)
            _, np_t_1, p_t_1, a = self.encode_board(new_board, None)
            next_state = self.tail.forward(np_t_1, p_t_1)
            next_states.append(next_state)
            actions_next.append(self.actor_target(next_state).detach())

        # rewards are always 0 except for terminal state +1/-1

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        Q_targets_next = []
        Q_expected = []

        for state, next_state, proto_action, action_next in zip(states, next_states, proto_actions, actions_next):
            Q_targets_next.append(self.critic_target(next_state, action_next).detach())
            # Compute critic loss
            if self.is_cuda:
                proto_action = proto_action.cuda()
                Q_expected.append(self.critic(state, proto_action))
        Q_targets_next = torch.cat(Q_targets_next, dim=0)
        Q_expected = torch.cat(Q_expected, dim=0)
        if self.is_cuda:
            dones = dones.cuda()
            rewards = rewards.cuda()

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.last_critic_loss = critic_loss.item()
        self.num_train_cycles += 1

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss

        actor_losses = []
        for state in states:
            state = state.detach()
            action_pred = self.actor(state)
            actor_losses.append(-self.critic(state, action_pred))
        actor_loss = torch.mean(torch.stack(actor_losses))
        self.last_actor_loss = actor_loss.item()
        # # Minimize the loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def postgame_update(self, replay_buffer, batch_size, updates=100):
        """
        Args:
            replay_buffer (dict):
            batch_size (int):
            updates (int):

        Returns:

        """
        perf.log("in update")
        critic_losses = {}
        actor_losses = {}
        with self.train_ctx():
            for i in range(updates):
                results = []
                batch_keys = random.sample(list(replay_buffer), k=batch_size)
                reps = []
                raw_actions = []
                perf.log("about to reconstitute boards and proto_actions")
                for step in batch_keys:

                    if len(step) > 1:
                        step = tuple(step)

                    obs, actions, proto_actions, result, new_obs = replay_buffer[step]

                    board = kf.Board(obs, self.board.configuration)
                    new_board = kf.Board(new_obs, self.board.configuration)

                    # non-plan and plan inputs at time t along with proto-actions
                    _, np_t, p_t, proto_actions_t = self.encode_board(board, proto_actions)
                    rep = self.tail.forward(np_t, p_t)

                    if proto_actions.nelement() == 0:
                        logger.critical(f"Got empty proto_actions in step {step}: {proto_actions.shape}")
                    if not board.current_player.shipyards:
                        logger.critical(f"Got no (old) shipyards in step {step}: {board.current_player.shipyards}")
                    if not new_board.current_player.shipyards:
                        logger.critical(f"Got no (new) shipyards in step {step}: {new_board.current_player.shipyards}")

                    reps.append(rep)
                    raw_actions.append(proto_actions_t)

                perf.log("done reconstituting, about to compute qs")

                reps = torch.cat(reps)
                raw_actions = torch.cat(raw_actions)
                qs = 0.5 * self.critic.forward(reps, raw_actions) + 0.5

                perf.log("done with update loop, about to compute final loss and backprop")

                ys = torch.ones_like(qs).to(qs.device) * torch.tensor(result).to(qs.device)
                results.append(result)
                if not (ys - qs).isfinite().all():
                    self.log_critical(f'Got non-finite in critic_loss: ys={ys}, qs={qs}')
                critic_loss = F.binary_cross_entropy(qs, ys)
                critic_loss.backward(retain_graph=True)
                critic_losses[i] = critic_loss.item()
                self.critic_opt.step()
                self.critic_opt.zero_grad()
                self.actor_opt.zero_grad()

                perf.log("done updating critic")

                policy = self.actor.forward(reps)
                actor_loss = self.critic.forward(reps, policy).mean()
                if not actor_loss.isfinite().all():
                    self.log_critical(f'Got non-finite in actor_loss: {actor_loss}')
                actor_loss.backward()
                actor_losses[i] = actor_loss.item()
                self.actor_opt.step()

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                self.log_error(
                    f"  results={results}, critic_loss={critic_loss.item():.6f}, actor_loss={actor_loss.item():.6f},"
                    f" qs={qs}, turns={batch_keys}",
                    every=self.log_losses_every
                )
                perf.log("done updating actor")
        perf.log("done updating")
        return actor_losses, critic_losses

    # noinspection PyMethodMayBeStatic
    def num_from_raw_action(self, raw_action, shipyard):
        num = max(1, -int(-raw_action[12] * shipyard.ship_count))
        if num > shipyard.ship_count:
            logging.warning(
                f"Trying to launch more ships than we have:{num} > {shipyard.ship_count}"
            )
            num = shipyard.ship_count
        return num

    # noinspection PyMethodMayBeStatic
    def plan_from_raw_action(self, raw_action, num_ships):
        return utils.PlanToDinger.dinger_to_plan(raw_action[13:].clone(), num_ships, apply_softmax=False, temperature=1.0)

    # noinspection PyMethodMayBeStatic
    def proto_to_raw(self, proto):
        raw_no_plan = proto[:, 0:12]
        raw_num = proto[:, 12:13]
        raw_plan = proto[:, 13:]
        raw_no_plan = torch.sigmoid(raw_no_plan)
        raw_num = torch.sigmoid(raw_num)
        raw_action = torch.cat([raw_no_plan, raw_num, raw_plan], dim=1)
        return raw_action

    def generate_exploratory_actions(self, proto_action, rep):
        k = 100  # self.num_explore_steps
        # step_size = (self.explore_sigma_max - self.explore_sigma_min) / k
        max_val = self.critic.forward(rep, proto_action)
        max_action = self.proto_to_raw(proto_action)

        for _ in range(0, k):
            sigma = torch.tensor(0.5)  # torch.tensor(self.explore_sigma_min + step_size * k)
            sample_amount = (1,)
            if self.is_cuda:
                sigma = sigma.cuda()
            new_action = tdist.Normal(proto_action, sigma)
            new_action = new_action.sample(sample_amount)
            new_action = new_action.squeeze(0)

            # convert new_action from proto_action to raw_action
            new_action = self.proto_to_raw(new_action)

            q_val = self.critic.forward(rep, new_action)

            if q_val > max_val:
                max_val = q_val
                max_action = new_action

        return max_action

    def should_random(self):
        return np.random.rand(1)[0] < self.epsilon

    def select_actions(self) -> Dict[str, str]:
        # nonplans dims = N x 12
        # plans dims = N x (1 + 256)
        shipyards = self.me.shipyards
        if not shipyards:
            self.last_pol = None
            return self.board.current_player.next_actions
        perf.log()
        nonplans, plans = self.encoded_board

        # rep dims 1 x 256 x 21 x 21
        perf.log("before_tail")
        rep = self.tail.forward(nonplans, plans)
        perf.log("after_tail")
        if self.explore:
            proto_action = self.actor.raw_forward(rep)
            raw_action = self.generate_exploratory_actions(proto_action, rep)
        else:
            raw_action = self.actor.forward(rep)

        # ONLY FOR DEBUGGING CRITIC
        # print(f"CRITIC = {self.critic(rep, raw_action)}")

        perf.log("after actor")
        self.last_pol = raw_action.detach()
        kore = self.kore
        spawn_cost = self.board.configuration.spawn_cost
        turn = self.board.step

        self.kore_spent_this_turn = 0
        self.ships_spawned_this_turn = 0
        self.ships_launched_this_turn = 0

        msg = (
            f"Turn {turn:3}: {self.kore:6.1f} kore, "
            f"yards {sum(sy.ship_count for sy in shipyards):3}/{len(shipyards):2}, "
            f"fleets {sum(f.ship_count for f in self.me.fleets):3}/{len(self.me.fleets):2}/{sum(f.kore for f in self.me.fleets):5.1f}."
        )
        perf.log(f"about to select action from e.g. {raw_action[0, :12, 0, 0]}")
        for shipyard in shipyards:
            r, c = utils.get_rc(shipyard)
            # act: (NUM_PLAN_TYPES + 23, )
            raw_act = raw_action[0, :, r, c]
            p_act = raw_act[:12].clone()
            max_spawn = min(int(kore / spawn_cost), shipyard.max_spawn)
            if max_spawn < utils.MAX_SPAWN:
                p_act[1 + max_spawn: 1 + utils.MAX_SPAWN] = 0.
            if shipyard.ship_count == 0:
                p_act[1 + utils.MAX_SPAWN:] = 0.
            p_act /= p_act.sum()  # re-normalize for entries that we zeroed out
            valid_choices = (p_act > 0.).double()
            try:
                if valid_choices.sum() >= 2 and self.should_random():
                    valid_choices[0] = 0.  # don't do nothing
                    choice = torch.multinomial(valid_choices, 1).item()  # choose one from the probability distribution
                    self.log_info(
                        f"Random choice (turn {turn})! {choice} from {(p_act > 0.).double()}",
                        every=20
                    )
                else:
                    choice = torch.multinomial(p_act, 1).item()  # choose one from the probability distribution
            except RuntimeError:
                self.log_critical(p_act)
                raise
            self.log_info(
                msg + f' For shipyard {shipyard.id} with {shipyard.ship_count} ships,'
                      f' got choice {choice} from {len(p_act)}, {p_act}',
                every=279
            )
            if choice == 0:
                if turn % self.log_stats_every == 0:
                    self.log_warning(msg + " Doing nothing.")
                else:
                    self.log_info(msg + " Doing nothing.")
            elif choice < 1 + utils.MAX_SPAWN:
                # spawn N ships = choice
                n = choice
                shipyard.next_action = kf.ShipyardAction.spawn_ships(n)
                kore -= spawn_cost * n
                self.kore_spent_this_turn += spawn_cost * n
                self.ships_spawned_this_turn += n
                if turn % self.log_stats_every == 0:
                    self.log_warning(msg + f' Spawn: {n}')
                else:
                    self.log_info(msg + f' Spawn: {n}')
            else:
                if self.should_random():
                    num = max(1, -int(-np.random.rand(1)[0] * shipyard.ship_count))
                    r_act = torch.rand_like(raw_act)
                    plan = self.plan_from_raw_action(r_act, num)
                    max_plan = kf.Fleet.max_flight_plan_len_for_ship_count(num)
                    self.log_error(
                        f"Random launch (turn {turn})! {num}@{plan} -> {plan[:max_plan]}",
                        every=20
                    )
                else:
                    num = self.num_from_raw_action(raw_act, shipyard)
                    max_plan = kf.Fleet.max_flight_plan_len_for_ship_count(num)
                    plan = self.plan_from_raw_action(raw_act, num)
                plan = plan[:max_plan]
                self.log_warning(msg + f' Launching ({shipyard.id}): {num:2}/{plan}.', every=5)
                shipyard.next_action = kf.ShipyardAction.launch_fleet_with_flight_plan(num, plan)
                self.ships_launched_this_turn += num
        perf.log("done selecting actions")
        return self.board.current_player.next_actions

    def generate_actions(
            self,
            obs: Dict[str, Any] = None,
            config: Union[kf.Configuration, Dict[str, Any]] = None
    ) -> Dict[str, str]:
        perf.log()
        self.board = kf.Board(obs, config)
        perf.log()
        return self.select_actions()


player = None


def agent(
        obs: Dict[str, Any],
        config: Union[kf.Configuration, Dict[str, Any]],
):
    global player
    if player is None:
        player = HeimerdingerAgent()
    return player.generate_actions(obs, config)
