import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from stable_baselines3.common.evaluation import evaluate_policy
from libs.target_state import TargetState, GeneralTargetState
from libs.agent import QuantumnAgent


class Benchmark:
    """
    Benchmarking utility for evaluating trained quantum state preparation agents
    across different suites of target states (TargetState and GeneralTargetState).

    Attributes:
        agent_list (list[QuantumnAgent]): The trained agents to benchmark.
        suites (dict): Dictionary of benchmark suites, each suite is a list of target states.
        results (pd.DataFrame): Stores evaluation results after running benchmarks.
    """

    def __init__(self, agent_list: list[QuantumnAgent]):
        """
        Initializes the Benchmark with a list of agents.

        Args:
            agent_list (list): A list of trained agents (in testing mode).
        """
        self.agent_list = agent_list
        self.suites = {}
        self.results = pd.DataFrame()

    def add_suite(self, suite_name: str, target_states: list[Union[TargetState, GeneralTargetState]]):
        """
        Add a suite of states to benchmark against.

        Args:
            suite_name (str): The name of the suite (e.g., 'fixed', 'scalable').
            target_states (list): A list of TargetState or GeneralTargetState objects.
        """
        self.suites[suite_name] = target_states

    def run_suite(self, suite_name: str, n_eval_episodes: int = 50):
        """
        Run benchmarking on a specific suite for all agents.

        Args:
            suite_name (str): Name of the suite to run.
            n_eval_episodes (int): Number of evaluation episodes per target state.
        """
        if suite_name not in self.suites:
            raise ValueError(f"Suite '{suite_name}' not found. Use add_suite first.")

        results = []
        for agent in self.agent_list:
            for ts in self.suites[suite_name]:
                agent.initialize_environment([ts])
                mean_reward, std_reward = evaluate_policy(
                    agent.model,
                    agent.env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True
                )
                results.append({
                    "agent": agent.model_folder_name,
                    "suite": suite_name,
                    "state": ts.state_name,
                    "num_qubits": ts.num_qubits,
                    "max_gates": ts.max_gates,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward
                })

        df = pd.DataFrame(results)
        self.results = pd.concat([self.results, df], ignore_index=True)
        return df

    def run_all(self, n_eval_episodes: int = 50):
        """
        Run benchmarking on all suites for all agents.

        Args:
            n_eval_episodes (int): Number of evaluation episodes per target state.
        """
        all_dfs = []
        for suite in self.suites.keys():
            all_dfs.append(self.run_suite(suite, n_eval_episodes))
        self.save_plot()
        return pd.concat(all_dfs, ignore_index=True)

    def save_plot(self, path: str = "./benchmark_plot.png"):
        """
        Save a comparison plot of all agents' performance.

        Args:
            path (str): Path to save the plot image.
        """
        if self.results.empty:
            print("No results to plot. Run benchmarking first.")
            return

        plt.figure(figsize=(10, 6))
        for agent_name, df_agent in self.results.groupby("agent"):
            plt.plot(
                df_agent["state"],
                df_agent["mean_reward"],
                marker="o",
                label=agent_name
            )

        plt.xlabel("Target State")
        plt.ylabel("Mean Reward")
        plt.title("Agent Benchmark Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        print(f"Plot saved to {path}")
