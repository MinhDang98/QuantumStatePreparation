import random
import math
import gymnasium as gym
from collections import deque
from libs.quantum_state_preparation import QuantumStatePreparation

class ALPBandTeacher:
	"""
	Compact ALP (Absolute Learning Progress) band teacher.
	- target_states_list: list of TargetState / GeneralTargetState objects
	- n_bins: number of discrete difficulty bands
	- window_size: per-bin sliding window size (ALP uses last window vs previous window)
	- replay_prob: probability to sample from replay buffer instead of newly sampled bin
	- difficulty_fn: function(ts) -> float, higher means harder. Default: ts.num_qubits * 100 + ts.max_gates
	"""
	def __init__(self,
				 target_states_list,
				 n_bins: int = 5,
				 window_size: int = 100,
				 replay_prob: float = 0.15,
				 difficulty_fn=None,
				 min_samples_for_alp:int=10):
		"""
		Initializes the ALPBandTeacher.

		Args:
			target_states_list: A list of target quantum states.
			n_bins (int): The number of discrete difficulty bands.
			window_size (int): The per-bin sliding window size.
			replay_prob (float): The probability of sampling from the replay buffer.
			difficulty_fn: A function to calculate the difficulty score of a state.
			min_samples_for_alp (int): The minimum number of samples needed to calculate ALP.
		"""
		self.target_states_list = list(target_states_list)
		self.n_bins = max(1, n_bins)
		self.window_size = max(1, window_size)
		self.replay_prob = float(replay_prob)
		self.min_samples_for_alp = min_samples_for_alp

		if difficulty_fn is None:
			# Default difficulty: emphasize number of qubits then max_gates
			def difficulty_fn_default(ts):
				return getattr(ts, "num_qubits", 1) * 100.0 + getattr(ts, "max_gates", 0)
			self.difficulty_fn = difficulty_fn_default
		else:
			self.difficulty_fn = difficulty_fn

		# Compute difficulty scores and bin indices
		scores = [(i, self.difficulty_fn(ts)) for i, ts in enumerate(self.target_states_list)]
		scores.sort(key=lambda x: x[1])
		indices_sorted = [i for i, _ in scores]

		# Chunk sorted indices into bins (as evenly as possible)
		self.bins = {b: [] for b in range(self.n_bins)}
		for idx_pos, global_idx in enumerate(indices_sorted):
			bin_idx = int(self.n_bins * idx_pos / max(1, len(indices_sorted)))
			# Clamp
			bin_idx = min(self.n_bins - 1, max(0, bin_idx))
			self.bins[bin_idx].append(global_idx)

		# Per-bin reward history (2*window to compare two windows)
		self.bin_rewards = {b: deque(maxlen=2 * self.window_size) for b in range(self.n_bins)}
		# Replay buffer stores global indices of interesting tasks (recent successes / high ALP)
		self.replay_buffer = deque(maxlen=1000)

	def _bin_alp(self, bin_idx):
		"""
		Compute ALP for a bin as absolute difference between mean(last window) and mean(prev window).
		If not enough samples, returns small epsilon.
		"""
		dq = self.bin_rewards[bin_idx]
		n = len(dq)
		if n < self.min_samples_for_alp:
			return 0.0
		half = min(self.window_size, n // 2)
		if half < 1:
			return 0.0
		recent = list(dq)[-half:]
		prev = list(dq)[-2*half:-half]
		if len(prev) < 1:
			return 0.0
		return abs(sum(recent)/len(recent) - sum(prev)/len(prev))

	def sample_bin(self):
		"""
		Sample a bin with probability proportional to ALP + epsilon (to encourage exploration).
		"""
		alps = []
		eps = 1e-6
		for b in range(self.n_bins):
			alp = self._bin_alp(b)
			alps.append(max(alp, eps))
		total = sum(alps)
		if total <= 0:
			# Uniform fallback
			probs = [1.0 / self.n_bins] * self.n_bins
		else:
			probs = [a/total for a in alps]
		# Ensure some exploration: mix with uniform
		mix = 0.15
		probs = [(1-mix)*p + mix*(1.0/self.n_bins) for p in probs]
		# Sample
		r = random.random()
		cum = 0.0
		for b, p in enumerate(probs):
			cum += p
			if r <= cum:
				return b
		return self.n_bins - 1

	def sample_task(self):
		"""
		Returns the global index of the sampled target state.
		With probability replay_prob, draw from replay buffer.
		Otherwise sample a bin based on ALP and then a random element from that bin.
		"""
		# Replay
		if len(self.replay_buffer) > 0 and random.random() < self.replay_prob:
			return random.choice(list(self.replay_buffer))

		# Choose a bin
		bin_idx = self.sample_bin()
		candidates = self.bins.get(bin_idx, [])
		if not candidates:
			# Fallback: sample any global index
			return random.randrange(len(self.target_states_list))
		return random.choice(candidates)

	def update(self, global_idx: int, fidelity: float):
		"""
		Notify teacher that an episode for a task (global_idx) finished with final fidelity.
		We append fidelity into bin history and optionally add to replay_buffer when interesting.
		"""
		# Map global_idx to bin
		bin_idx = None
		for b, lst in self.bins.items():
			if global_idx in lst:
				bin_idx = b
				break
		if bin_idx is None:
			# Shouldn't happen, but place in last bin
			bin_idx = self.n_bins - 1
		self.bin_rewards[bin_idx].append(float(fidelity))

		# Heuristics for replay: if success or high improvement, add to replay
		# E.g., if fidelity > 0.95 or fidelity increased recently
		if fidelity >= 0.95:
			self.replay_buffer.append(global_idx)
		# Also keep some random successful tasks
		elif fidelity >= 0.8 and random.random() < 0.1:
			self.replay_buffer.append(global_idx)

	def get_target_by_index(self, global_idx: int):
		return self.target_states_list[global_idx]

	def debug_stats(self):
		# Return a compact dict for logging
		stats = {}
		for b in range(self.n_bins):
			dq = list(self.bin_rewards[b])
			stats[f"bin{b}_count"] = len(dq)
			stats[f"bin{b}_alp"] = self._bin_alp(b)
		stats["replay_size"] = len(self.replay_buffer)
		return stats

class TeacherEnvWrapper(gym.Env):
	"""
	A wrapper that holds a single QuantumStatePreparation but on each reset
	samples a task from the provided ALPBandTeacher.
	It reports fidelity to the teacher when episodes finish.
	This wrapper mirrors QuantumStatePreparation's API (Gymnasium 5-tuple).
	"""
	def __init__(self, target_states_list, teacher: ALPBandTeacher,
				 max_env_qubits: int, max_env_gates: int):
		# Do not call super().__init__() too strictly; this is a thin wrapper.
		self.target_states_list = target_states_list
		self.teacher = teacher
		self.max_env_qubits = max_env_qubits
		self.max_env_gates = max_env_gates

		# Underlying env can be initialized with the full list; we will pick a target on reset
		self.env = QuantumStatePreparation(
			target_states_list=self.target_states_list,
			max_env_qubits=self.max_env_qubits,
			max_env_gates=self.max_env_gates
		)
		# Expose space info to SB3
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.current_global_idx = None

	def reset(self, seed=None, **kwargs):
		"""
		Sample a task index from teacher and reset the underlying env accordingly.
		Returns the full (observation, info) tuple as expected by VecEnv.
		"""
		# Sample a task index from teacher
		global_idx = self.teacher.sample_task()
		self.current_global_idx = global_idx
		target_state_obj = self.target_states_list[global_idx]
  
		# Print(f"ALP Teacher selected state: {target_state_obj.state_name} ({target_state_obj.num_qubits} qubits)")
		
		# Call the underlying reset, passing the target state
		# The underlying QuantumStatePreparation.reset() returns (observation, info)
		observation, info = self.env.reset(seed=seed, target_state_object=target_state_obj)

		# Check if the observation is the correct type.
		# If it's not a dict, something is wrong with the underlying env.
		if not isinstance(observation, dict):
			raise RuntimeError(
			f"TeacherEnvWrapper.reset: underlying env returned unexpected observation type "
			f"{type(observation)}. Expected a dictionary. Offending value: {repr(observation)[:200]}"
			)

		return observation, info

	def step(self, action):
		# Forward to underlying env
		obs, reward, terminated, truncated, info = self.env.step(action)
		done = terminated or truncated

		if done:
			# Report final fidelity to teacher
			fid = info.get("fidelity", 0.0)
			if self.current_global_idx is not None:
				try:
					self.teacher.update(self.current_global_idx, fid)
				except Exception:
					pass
			self.current_global_idx = None

		return obs, reward, terminated, truncated, info

	def action_masks(self):
		"""Forward to underlying env's action mask."""
		return self.env.compute_valid_action_mask()
