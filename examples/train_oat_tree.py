# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Entry script of using OAT to RL-tune LLM agents on GEM environments.
"""
import functools
import json
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple
import random

import numpy as np
import torch
import tree
import vllm
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData
from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import Dataset

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns

""" +=========================================+ """
""" 1. Defining constants used in our training. """
""" +=========================================+ """

from examples.train_oat import (
    INVALID_ACTION,
    TEMPLATE_FACTORY,
)


""" +=================================================+ """
""" 2. Defining extra arguments/structure for training. """
""" +=================================================+ """


@dataclass
class Args(PPOArgs):
    # Environment settings
    env_id: str = "rg:leg_counting"
    num_env: int = 1
    wrappers: str = ""
    async_env: bool = False

    # Template settings
    prompt_template: Literal["qwen3_game", "no", "qwen3_general", "code"] = "qwen3_game"

    # Reward settings
    gamma: float = 1.0  # Discount factor for Monte Carlo returns
    norm_return: bool = False

    # Evaluation settings
    eval_prompt_template: Literal["qwen3_general"] = "qwen3_general"
    eval_data: Optional[str] = "./data"
    eval_input_key: str = "input"
    eval_output_key: str = "answer"
    eval_split: str = "all"

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Tree args
    max_branches: int = 8  # Max branches to explore per tree
    value_propagation_type: Literal["mean", "meanmax", "max"] = "meanmax"
    discard_zero_adv: bool = False  # Discard transitions with zero advantage


@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    done: bool

    prompt: str
    prompt_ids: list
    response: str
    response_ids: list
    response_logprobs: list

    response_is_truncated: bool
    action_is_formatted: bool

    returns: float = None  # To be filled in later
    advantage: float = None  # To be filled in later

    def format(self):
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }

    def __repr__(self):
        return f"Transition(obs={self.obs}, action={self.action}, reward={self.reward}, done={self.done})"


""" +=======================================+ """
""" 3. Defining actor to collect experiences. """
""" +=======================================+ """

class TreeNode:
    """Represents a single transition (s, a, r, s') in the episode collection tree."""
    def __init__(self,
                 env_in_current_state, env_obs,
                 transition: Transition, parent: Optional['TreeNode']
                 ):
        self.env_in_current_state = env_in_current_state
        self.env_obs = env_obs
        self.transition = transition
        
        self.parent = parent
        self.step_num = 0 if self.parent is None else self.parent.step_num + 1
        self.children: List['TreeNode'] = []
        
        # For value propagation
        self.return_for_transition_mean = None # G(s, a) = r + gamma * V_mean(s')
        self.return_for_transition_max = None  # G(s, a) = r + gamma * V_max(s')
        self.baseline_value_mean = None # V(s') = mean of return_for_transition_mean of all siblings
        self.baseline_value_max = None # V(s') = mean of return_for_transition_max of all siblings
        self.advantage = None
        self.undiscounted_return = None # G(s, a) = r + V_mean(s') # for logging only

    def add_child(self, child_node: 'TreeNode'):
        """Adds a child node to this node."""
        self.children.append(child_node)

    def delete_env(self):
        """Deletes the stored environment to save memory."""
        del self.env_in_current_state
        self.env_in_current_state = None
        del self.env_obs
        self.env_obs = None

    def __repr__(self):
        # A helper for printing and debugging
        string = f"Node({self.transition=}, {len(self.children)=}"
        string += f"ret_mean={self.return_for_transition_mean}, "
        string += f"ret_max={self.return_for_transition_max}, "
        string += f"base_mean={self.baseline_value_mean}, "
        string += f"base_max={self.baseline_value_max}, "
        string += f"advantage={self.advantage}"

class Tree:
    def __init__(self,
                 env_in_current_state,
                 env_obs,
                 max_branches: int,
                 gamma: float,
                 value_propagation_type: str = "mean", # or "meanmax" or "max"
                 ):
        self.max_branches = max_branches
        self.branch_counter = 0
        self.gamma = gamma
        self.value_propagation_type = value_propagation_type
        self.all_nodes = []
        self.current_node = TreeNode(
            env_in_current_state=env_in_current_state,
            env_obs=env_obs,
            transition=None,
            parent=None
        )
        self.all_nodes.append(self.current_node)
        self.root_node: TreeNode
        self.root_node = self.all_nodes[0]

        self.total_dones = 0
        self.total_successes = 0
        self.lengths = []
        self.finished = False

    def get_total_transitions_collected(self):
        """Returns the total number of transitions (excluding the root).
        Includes all - not just the ones that are part of pairs."""
        return len(self.all_nodes) - 1 # exclude root

    def step(self, obs, action, reward, done, extra,
             env_in_current_state, env_obs):
        """Takes a step in the tree with the given transition."""
        if extra["generation_failed"]:
            self.current_node.transition.reward += reward
            self.current_node.transition.done = True
            done = True
        else:
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                prompt=extra["formatted_observation"],
                prompt_ids=extra["prompt_ids"],
                response=extra["response"],
                response_ids=extra["response_ids"],
                response_logprobs=extra["response_logprobs"],
                response_is_truncated=extra["response_is_truncated"],
                action_is_formatted=extra["action_is_formatted"],
            )
            new_node = TreeNode(
                env_in_current_state=env_in_current_state,
                env_obs=env_obs,
                transition=transition,
                parent=self.current_node
            )
            self.current_node.add_child(new_node)
            self.all_nodes.append(new_node)
        
        if done:
            self.branch_counter += 1
            self.total_dones += 1
            self.total_successes += 1 if reward == 1.0 else 0
            self.lengths.append(self.current_node.step_num if extra["generation_failed"] else new_node.step_num)
            if self.branch_counter >= self.max_branches:
                self.finished = True
            else:
                nodes_with_one_action = [node for node in self.all_nodes if len(node.children) == 1]
                if len(nodes_with_one_action) == 0:
                    self.finished = True

        if done and not self.finished:
            sampled_node = random.choice(nodes_with_one_action)
            self.current_node = sampled_node
            sampled_node: TreeNode
            set_env = True
            env = deepcopy(sampled_node.env_in_current_state)
            obs = deepcopy(sampled_node.env_obs)
            self.current_node.delete_env()
        else:
            self.current_node = new_node
            set_env = False
            env = None
            obs = None
        return set_env, env, obs, self.finished

    def propagate_values(self):
        """Propagates values from leaf nodes to the root."""
        node: TreeNode
        
        # Return for leaf nodes
        for node in self.all_nodes:
            node.delete_env() # delete envs to save memory
            if len(node.children) == 0:
                assert node.transition.done, f"{node=}"
                node.return_for_transition_mean = node.transition.reward
                node.return_for_transition_max = node.transition.reward

        # Propagate returns backwards
        for node in reversed(self.all_nodes):
            if len(node.children) > 0:
                # Check since these values should not have been set yet
                assert all(x is None for x in [
                    node.return_for_transition_mean,
                    node.return_for_transition_max,
                    node.baseline_value_mean,
                    node.baseline_value_max,
                    ]), f"Node: {node} unexpectedly has values already set."
                # Set the children baselines and advantages
                children_baseline_mean = np.mean([child.return_for_transition_mean for child in node.children])
                children_baseline_max = np.mean([child.return_for_transition_max for child in node.children])
                for child in node.children:
                    assert child.baseline_value_mean is None
                    assert child.baseline_value_max is None
                    child.baseline_value_mean = children_baseline_mean
                    child.baseline_value_max = children_baseline_max
                    if self.value_propagation_type == "mean":
                        child.advantage = child.return_for_transition_mean - children_baseline_mean
                    elif self.value_propagation_type == "meanmax":
                        child.advantage = child.return_for_transition_max - children_baseline_mean
                    elif self.value_propagation_type == "max":
                        child.advantage = child.return_for_transition_max - children_baseline_max
                if node.parent is not None: # skip root
                    assert node.transition.done == False, f"Non-leaf node has done transition: {node=}"
                    # Set the return
                    node.return_for_transition_mean = node.transition.reward + self.gamma * children_baseline_mean
                    node.return_for_transition_max = node.transition.reward + self.gamma * children_baseline_max
                    node.undiscounted_return = node.transition.reward + children_baseline_mean

        # Collect all nodes that are one of a pair of siblings and have non-zero advantage
        self.pair_transitions_nonzero_adv = []
        self.pair_transitions_zero_adv = []
        for node in self.all_nodes[1:]: # skip root
            if len(node.parent.children) == 1:
                assert node.advantage == 0, f"Node has no siblings but non-zero advantage. This shouldn't happen: {node=}, {node.advantage}"
            elif len(node.parent.children) == 2:
                # if node.advantage != 0.0:
                node.transition.advantage = node.advantage
                node.transition.returns = node.return_for_transition_mean
                if node.advantage == 0.0:
                    self.pair_transitions_zero_adv.append(node.transition)
                else:
                    self.pair_transitions_nonzero_adv.append(node.transition)
            else:
                raise ValueError(f"Unexpected number of children: {node.parent=}")

        # Set root node/tree mean total undiscounted return for logging
        self.root_node.undiscounted_return = np.mean(
            [child.return_for_transition_mean for child in self.root_node.children]
        )

        self.values_propagated = True


class Actor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.args.seed += 233 ** (actor_id + 1)
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        if actor_id == 0:
            os.makedirs(self.game_state_save_path, exist_ok=True)
        self.args: Args = self.args
        args = self.args
        self.oracle = None

        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=1,
            logprobs=True,
        )

        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,
            logprobs=True,
        )

        self.step_count = 0

        # Get environment wrappers.
        wrappers = get_wrapper_fns(self.args.wrappers, tokenizer=self.tokenizer)

        # Instantiate vectorized environment.
        self.env = gem.make_vec(
            [self.args.env_id] * self.args.num_env,
            vec_kwargs=[{"seed": self.args.seed + j} for j in range(self.args.num_env)],
            wrappers=wrappers,
            async_mode=self.args.async_env,
        )

    def step(
        self, prompts=None, formatted_prompts=None, references=None
    ) -> List[TransitionData]:
        """Each actor.step handles the interaction between agent and environment to collect experiences."""
        # The provided parameters are ignored since we generate prompts from the environment
        del prompts, formatted_prompts, references

        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
        info = {}

        # Play multiple episodes to generate transitions (trajectories in language MDP)
        all_transitions = []

        transitions, collection_info = self.collect_experience(
            self.env, self.args.rollout_batch_size_per_device
        )
        all_transitions.extend(self.prepare_trajectories(transitions))

        # logging infos
        info["actor/num_transitions"] = len(all_transitions)

        # update collection info
        info.update(
            {k.replace("actor/", "actor/"): v for k, v in collection_info.items()}
        )

        # Subsample transitions if they exceed the batch size
        if len(all_transitions) > self.args.rollout_batch_size_per_device:
            subsample_indices = np.random.choice(
                len(all_transitions),
                self.args.rollout_batch_size_per_device,
                replace=False,
            )
            all_transitions = [all_transitions[si] for si in subsample_indices]
        logging.info(f"Actor finished collecting {len(all_transitions)} transitions")

        for transition in all_transitions:
            transition.info.update(**info)

        self.step_count += 1
        # Serialize and return the transitions
        handle = self.ipc_client.serialize_ipc(all_transitions)
        return handle  # type: ignore

    def collect_experience(self, env, min_steps: int):
        tree: Tree
        total_pair_transitions_nonzero_adv = 0
        total_pair_transitions_zero_adv = 0
        num_generation_failed = 0
        num_generations = 0

        obs, _ = env.reset()
        trees = [Tree(
            env_in_current_state=deepcopy(env.envs[i]),
            env_obs=deepcopy(obs[i]),
            max_branches=self.args.max_branches,
            gamma=self.args.gamma,
            value_propagation_type=self.args.value_propagation_type,
        ) for i in range(env.num_envs)]
        finished_trees = []
        while True:
            action, extra = self.agent_act(obs)  # type: ignore
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):
                num_generations += 1
                num_generation_failed += 1 if extra[i]["generation_failed"] else 0
                set_env, new_env, new_obs, tree_finished = trees[i].step(
                    obs=obs[i],
                    action=action[i],
                    reward=reward[i],
                    done=done[i],
                    extra=extra[i],
                    env_in_current_state=deepcopy(env.envs[i]) if not done[i] else None,
                    env_obs=deepcopy(next_obs[i]) if not done[i] else None,
                )
                if set_env:
                    env.envs[i] = new_env
                    next_obs[i] = new_obs
                if tree_finished:
                    trees[i].propagate_values()
                    finished_trees.append(trees[i])
                    total_pair_transitions_nonzero_adv += len(trees[i].pair_transitions_nonzero_adv)
                    total_pair_transitions_zero_adv += len(trees[i].pair_transitions_zero_adv)
                    next_obs[i] = env.envs[i].reset()[0]
                    trees[i] = Tree(
                        env_in_current_state=deepcopy(env.envs[i]),
                        env_obs=deepcopy(next_obs[i]),
                        max_branches=self.args.max_branches,
                        gamma=self.args.gamma,
                        value_propagation_type=self.args.value_propagation_type,
                    )

            # Check if collected enough transitions
            too_many_zero_adv = False
            if self.args.discard_zero_adv:
                if total_pair_transitions_nonzero_adv >= min_steps:
                    break
                elif total_pair_transitions_zero_adv >= min_steps*2:
                    # If we have too many zero-advantage transitions, we stop to avoid wasting time
                    too_many_zero_adv = True
                    logging.warning(f"Actor-{self.actor_id} stopping collection early due to too many zero-advantage transitions.")
                    break
            else:
                if (total_pair_transitions_nonzero_adv + total_pair_transitions_zero_adv) >= min_steps:
                    break

        # Collect transitions from all finished trees
        pair_transitions_nonzero_adv = []
        pair_transitions_zero_adv = []
        for tree in finished_trees:
            assert tree.values_propagated
            pair_transitions_nonzero_adv.extend(tree.pair_transitions_nonzero_adv)
            pair_transitions_zero_adv.extend(tree.pair_transitions_zero_adv)
        assert len(pair_transitions_nonzero_adv) == total_pair_transitions_nonzero_adv
        assert len(pair_transitions_zero_adv) == total_pair_transitions_zero_adv
        num_zero_needed = max(0, min_steps - len(pair_transitions_nonzero_adv))
        transitions = pair_transitions_nonzero_adv + pair_transitions_zero_adv[:num_zero_needed]
        assert len(transitions) >= min_steps, f"Actor-{self.actor_id} did not collect enough transitions. Shouldn't happen: {len(pair_transitions_nonzero_adv)=}, {len(pair_transitions_zero_adv)=}, {len(transitions)=}, {min_steps=}, {too_many_zero_adv=}"

        # For logging
        mean_episode_return = 0.0
        total_transitions_collected = 0
        for tree in finished_trees:
            mean_episode_return += tree.root_node.undiscounted_return
            total_transitions_collected += tree.get_total_transitions_collected()
        mean_episode_return /= len(finished_trees)
        prop_total_transitions_wasted = 1.0 - len(transitions) / total_transitions_collected
        prop_pair_transitions_wasted = 1.0 - len(transitions) / (total_pair_transitions_nonzero_adv + total_pair_transitions_zero_adv)
        prop_zero_adv = np.mean([1.0 if t.advantage == 0.0 else 0.0 for t in transitions])

        # info
        total_tree_size = 0
        total_successes = 0
        total_dones = 0
        lengths = []
        for tree in finished_trees:
            total_tree_size += len(tree.all_nodes) - 1
            total_successes += tree.total_successes
            total_dones += tree.total_dones
            lengths.extend(tree.lengths)
        
        info = {
            "actor/num_trees": len(finished_trees),
            "actor/total_tree_size": total_tree_size,
            "actor/avg_tree_size": total_tree_size / len(finished_trees),
            "actor/total_transitions_collected_per_tree": total_transitions_collected / len(finished_trees),
            "actor/total_transitions": len(transitions),
            "actor/avg_transitions_per_tree": len(transitions) / len(finished_trees),
            "actor/total_successes": total_successes,
            "actor/total_dones": total_dones,
            "actor/mean_episode_success": total_successes / total_dones,
            "actor/mean_episode_length": np.mean(lengths),
            "actor/max_episode_length": np.max(lengths),
            "actor/min_episode_length": np.min(lengths),
            "actor/mean_episode_return": mean_episode_return,
            "actor/total_transitions_collected": total_transitions_collected,
            "actor/prop_total_transitions_wasted": prop_total_transitions_wasted,
            "actor/prop_pair_transitions_wasted": prop_pair_transitions_wasted,
            "actor/prop_zero_adv": prop_zero_adv,
            "actor/too_many_zero_adv": float(too_many_zero_adv),
            "actor/num_generations": num_generations,
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": num_generation_failed / num_generations,
        }

        return transitions, info

    def agent_act(self, vec_observation: List[str]) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.

        """
        formatted_observations = []
        for observation in vec_observation:
            observation = TEMPLATE_FACTORY[self.args.prompt_template](observation)
            if self.args.apply_chat_template:
                observation = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_observations.append(observation)

        sampling_params = (
            self.eval_sampling_params if self.eval_mode else self.sampling_params
        )

        # Subsample to remove observations that exceed max model length
        idss = self.tokenizer(formatted_observations).input_ids
        exceeds_lengths = [len(ids) >= self.args.max_model_len for ids in idss]
        sub_formatted_observations = [
            o for o, e in zip(formatted_observations, exceeds_lengths) if not e
        ]

        # Generate
        sub_outputs = self.generate(sub_formatted_observations, sampling_params)

        executable_actions = []
        extras = []
        sub_i = 0
        for i, exceeds_length in enumerate(exceeds_lengths):
            if exceeds_length:
                # if prompt exceeds max model length we skipped the generation
                executable_actions.append(INVALID_ACTION)
                extras.append({"generation_failed": True})
            else:
                raw_action = sub_outputs[sub_i].outputs[0].text
                prompt_token_ids = sub_outputs[sub_i].prompt_token_ids
                token_ids = sub_outputs[sub_i].outputs[0].token_ids
                response_logprobs = sub_outputs[sub_i].outputs[0].logprobs
                response_logprobs = [
                    item[token_ids[i]].logprob
                    for i, item in enumerate(response_logprobs)
                ]
                response_is_truncated = (
                    sub_outputs[sub_i].outputs[0].finish_reason == "length"
                )

                # Valid extraction = proper eos + proper format
                # Only used for metric logging
                extracted_action = (
                    INVALID_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action)
                )
                executable_actions.append(
                    INVALID_ACTION if response_is_truncated else raw_action
                )
                extras.append(
                    {
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_token_ids,
                        "response": raw_action,
                        "response_ids": token_ids,
                        "response_logprobs": response_logprobs,
                        "response_is_truncated": response_is_truncated,
                        "action_is_formatted": extracted_action != INVALID_ACTION,
                        "generation_failed": False,
                        "generation_max_length_reached": (
                            len(prompt_token_ids) + len(token_ids)
                            >= self.args.max_model_len
                        ),
                    }
                )
                sub_i += 1
        return executable_actions, extras  # type: ignore

    def prepare_trajectories(
        self, transitions: List[Transition]
    ) -> List[TransitionData]:
        """
        Prepare Transitions into TrajectoryData for OAT.
        "Trajectory" here refers to a sequence of tokens, ie. text.
        """
        trajectory_data = []
        # Distribute turn-based returns to token-level returns
        for transition in transitions:
            transition: Transition
            dense_rewards = self.compute_token_level_rewards(
                transition.response_ids, transition.advantage
            )
            # Add trajectory data
            trajectory_data.append(
                TransitionData(
                    prompt=transition.prompt,
                    prompt_ids=transition.prompt_ids,
                    response=transition.response,
                    response_ids=transition.response_ids,
                    # response_logprobs=None,  # Re-calculated on learner side.
                    response_logprobs=transition.response_logprobs,
                    rewards=dense_rewards,
                    loss_mask=(
                        not transition.response_is_truncated
                        if self.args.ignore_no_eos
                        else True
                    ),
                    info={
                        "actor/action_is_formatted": transition.action_is_formatted,
                        "actor/step_reward": transition.reward,
                        "actor/discount_factor": self.args.gamma,
                        "actor/discounted_step_return": transition.returns,
                        "actor/advantage": transition.advantage,
                        "actor/response_is_truncated": transition.response_is_truncated,
                        "actor/timestamp": time.time_ns(),
                    },
                )
            )
        return trajectory_data

    def compute_token_level_rewards(
        self, token_ids: List[int], discounted_reward: float
    ) -> List[float]:
        # Initialize all tokens with zero reward
        dense_rewards = [0.0] * len(token_ids)
        # Last token gets full discounted reward
        dense_rewards[-1] = discounted_reward
        return dense_rewards

    def extract_action(self, text: str) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            formatted_action = None
            if self.args.prompt_template in ["qwen3_game", "qwen3_general"] or (
                self.args.prompt_template == "no"
                and "qwen" in self.args.pretrain.lower()
            ):
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif self.args.prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                raise NotImplementedError

            if formatted_action is None:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy OAT's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return "", "", ""


""" +====================================+ """
""" 4. Defining learner update the policy. """
""" +====================================+ """


class Learner(PPOLearner):
    def _init(self, args: Args, actors: List[Actor]) -> None:
        """
        Initialize the learner.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

    def prepare_data(self, strategy, tokenizer):
        """
        Override the data preparation to avoid loading external datasets.
        Instead, create dummy datasets just to keep OAT's infrastructure happy.
        """
        # Create dummy dataset that satisfies OAT's requirements
        # but doesn't actually load any data
        # Used to control the training episode, set a large number.
        self.prompts_dataset = DummyPromptDataset(size=int(1e9))

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )

    def process_feedback_data(self, data_list: List[TransitionData]):
        """Process collected feedback data, adding it to buffer."""

        logging.info("adding data into buffer")

        # Add to buffer
        self.pi_buffer.extend(data_list)

        # Also add to all_buffer if we're tracking all data
        if self.args.dump_all_buffer:
            self.all_buffer.extend(data_list)

        # Update query step (for tracking progress)
        self.query_step += len(data_list)

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        # Return without baseline
        advantages = rewards.sum(-1)
        if self.args.norm_return:
            local_sum = advantages.sum()
            local_square_sum = (advantages**2).sum()
            local_num = torch.tensor(
                [advantages.numel()], dtype=torch.float32, device=advantages.device
            )

            global_sum = self.strategy.all_reduce(local_sum, op="sum")
            global_square_sum = self.strategy.all_reduce(local_square_sum, op="sum")
            global_num = self.strategy.all_reduce(local_num, op="sum")

            mean_adv = global_sum / global_num
            std_adv = torch.sqrt(global_square_sum / global_num - mean_adv**2)
            advantages = (advantages - mean_adv) / (std_adv + 1e-9)
        return advantages


def train(args: Args):
    """
    Reinforcement learning starts here.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(args, learner_cls=Learner, actor_cls=Actor)

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    # Get default arguments and customize them
    args: Args = get_default_args(Args)

    # Customization
    args.algo = "PPO"
    args.eval_batch_size = 32

    # CRITICAL: Disable oracle and dataset loading
    args.oracle = ""  # Empty string for no external oracle
    args.prompt_data = ""  # Don't load any dataset
    args.rollout_batch_size = args.rollout_batch_size_per_device * args.gpus
    if "concat_chat" in args.wrappers:
        assert (
            args.prompt_template == "no"
        ), "chat template is applied on env side already"
    args = default_args_validation(args)

    # Let's go
    train(args)
