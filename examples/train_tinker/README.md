## Training with Tinker and Tinker Cookbook

The [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) provides a practical framework for fine-tuning LLMs through [Tinker](https://thinkingmachines.ai/tinker/), an API that abstracts away the complexities of distributed training. Tinker manages backend training workflows, while the Cookbook delivers a suite of high-level recipes and utilities to accelerate experimentation for researchers, developers, and practitioners. 

In this guide, we present two implementations leveraging Tinker:
- Standard Training with Cookbook Utilities:
   We introduce an adapter layer that enables GEM to work seamlessly with the Tinker Cookbook’s training loop.
- Custom Training via Low-level API:
   For advanced use cases, we provide a basic RL implementation that interacts directly with Tinker’s low-level API, enabling maximum customization.


### Getting Started

Before running the experiments, install the required libraries as follows:
```bash
# install gem
pip install -U gem-llm

# install tinker
git clone git@github.com:thinking-machines-lab/tinker-cookbook.git
cd ./tinker-cookbook
pip install -e .

# export tinker api key
export TINKER_API_KEY=<your-tinker-api-key>
```

### Using Tinker Cookbook
To train agents with the Tinker Cookbook, we implemented an adaption layer (`tinker_cookbook_adapter.py`) that exposes a GEM-compatible interface. 

**Example 1: RL Training on Math Environments**

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=math:Math12K groups_per_batch=64 group_size=16 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base env_kwargs_json='{"use_mp": false}'
```

Note:
- You may train on different math environments by simply changing the `env_id` argument.
- `env_kwargs_json='{"use_mp": false}'` is only required for math environments.

**Example 2: Training on Reasoning Gym**

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=rg:simple_equations groups_per_batch=64 group_size=8 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base
```