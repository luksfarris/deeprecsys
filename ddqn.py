import torch
import numpy as np
from collections import namedtuple, deque
from copy import deepcopy
from random_agent import RandomAgent


class DDQN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_actions, learning_rate, device="cpu"):
        """
        Params
        ======
        n_inputs: tamaño del espacio de estadps
        n_outputs: tamaño del espacio de acciones
        actions: array de acciones posibles
        """
        super(DDQN, self).__init__()
        self.device = device
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.actions = n_actions
        self.learning_rate = learning_rate

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.n_outputs, bias=True),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.device == "cuda":
            self.model.cuda()

    def get_action(self, state, epsilon=0.05):
        # Método e-greedy
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action = torch.max(qvals, dim=-1)[1].item()
        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)


class experienceReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "done", "next_state"],
        )
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        if type(action) is np.ndarray:
            action = action.tolist()
        self.replay_memory.append(
            self.experience(state, action, reward, done, next_state)
        )

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class DDQNAgent(RandomAgent):
    def __init__(
        self,
        env,
        dnnetwork,
        buffer,
        epsilon=0.1,
        eps_decay=0.99,
        batch_size=32,
        gamma=0.99,
        max_episodes=50000,
        dnn_update_frequency=4,
        dnn_sync_frequency=2000,
        nblock=20,
        reward_threshold=1000,
        feature_transformer=None,
    ):
        """ "
        Params
        ======
        env: entorno
        dnnetwork: clase con la red neuronal diseñada
        target_network: red objetivo
        buffer: clase con el buffer de repetición de experiencias
        epsilon: epsilon
        eps_decay: epsilon decay
        batch_size: batch size
        nblock: bloque de los X últimos episodios de los que se calculará
             la media de recompensa
        reward_threshold: umbral de recompensa definido en el entorno
        """
        super().__init__(env, feature_transformer)
        self.env = env
        self.dnnetwork = dnnetwork
        self.target_network = deepcopy(dnnetwork)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.dnn_update_frequency = dnn_update_frequency
        self.dnn_sync_frequency = dnn_sync_frequency
        self.nblock = nblock
        self.reward_threshold = reward_threshold
        self.initialize()
        self.fill_buffer()

    def initialize(self):
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0

    def take_step(self, epsilon, state, mode="train") -> bool:
        if mode == "explore":
            slate_action = (
                self.env.action_space.sample()
            )  # acción aleatoria en el burn-in
        else:
            # acción a partir del valor de Q (elección de la acción con mejor Q)
            action = self.dnnetwork.get_action(state, epsilon)
            self.step_count += 1
            slate_action = [action]
        return slate_action

    def fill_buffer(self):
        # Rellenamos el buffer con N experiencias aleatorias ()
        # print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            state = self.env.reset()
            state = self.features_from_state(state, [])
            slate_action = self.take_step(self.epsilon, state, mode="explore")
            new_state, reward, done, _ = self.env.step(slate_action)
            new_state = self.features_from_state(new_state, slate_action)
            self.buffer.append(state, slate_action, reward, done, new_state)

    def get_next_action(self, state):
        # El agente toma una acción
        slate_action = self.take_step(self.epsilon, state, mode="train")
        # Actualizar la red principal según la frecuencia establecida
        if self.step_count % self.dnn_update_frequency == 0:
            self.update()
        # Sincronizar red principal y red objetivo según la frecuencia establecida
        if self.step_count % self.dnn_sync_frequency == 0:
            self.target_network.load_state_dict(self.dnnetwork.state_dict())
        return slate_action

    def episode_finished(self):
        # Actualizar epsilon según la velocidad de decaimiento fijada########
        self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    # Cálculo de la pérdida

    def calculate_loss(self, batch):
        # Separamos las variables de la experiencia y las convertimos a tensores
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = (
            torch.FloatTensor(rewards).to(device=self.dnnetwork.device).reshape(-1, 1)
        )
        actions_vals = (
            torch.LongTensor(np.array(actions))
            .reshape(-1, 1)
            .to(device=self.dnnetwork.device)
        )
        dones_t = torch.BoolTensor(dones).to(device=self.dnnetwork.device)

        # Obtenemos los valores de Q de la red principal
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)

        # DDQN update
        # Obtenemos la acción con máximo valor de Q de la red principal
        next_actions = [self.dnnetwork.get_action(s, 0) for s in next_states]
        next_actions_vals = (
            torch.LongTensor(next_actions)
            .reshape(-1, 1)
            .to(device=self.dnnetwork.device)
        )

        # Obtenemos los valores de Q de la red objetivo
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()

        qvals_next[dones_t] = 0

        # Calculamos ecuación de Bellman
        expected_qvals = self.gamma * qvals_next + rewards_vals

        # Calculamos la pérdida
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1, 1))
        return loss

    def update(self):
        self.dnnetwork.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        batch = self.buffer.sample_batch(
            batch_size=self.batch_size
        )  # seleccionamos un conjunto del buffer
        loss = self.calculate_loss(batch)  # calculamos la pérdida
        loss.backward()  # hacemos la diferencia para obtener los gradientes
        self.dnnetwork.optimizer.step()  # aplicamos los gradientes a la red neuronal


def create_ddqn_agent(env):
    n_inputs = 25  # env.observation_space.shape[0]
    n_outputs = env.action_space.nvec[0]  # env.action_space.n
    learning_rate = 25 * (10 ** -5)
    batch_size = 32
    memory_size = 10000
    burn_in = 1000
    start_epsilon = 1
    epsilon_decay = 0.99
    # gamma = 0.99
    # max_episodes = 3000
    # dnn_update_frequency = 5
    # dnn_sync_frequency = 1000

    ddqn = DDQN(n_inputs, n_outputs, np.arange(env.action_space.nvec[0]), learning_rate)
    erb = experienceReplayBuffer(memory_size, burn_in)
    agent = DDQNAgent(env, ddqn, erb, start_epsilon, epsilon_decay, batch_size)
    return agent
