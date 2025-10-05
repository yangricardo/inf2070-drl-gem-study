## Training with Tinker and Tinker Cookbook

[introduction]

Before you start the experiments, you could install the library using:

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

### Tinker Cookbook
To train agents with tinker cookbook directly, we implemented an adaption layer (`tinker_cookbook_adapter.py`) for GEM to expose the interface compatible with it. 

Below we show the example of RL training on **Math environments**: 

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=math:Math12K groups_per_batch=64 group_size=16 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base env_kwargs_json='{"use_mp": false}'
```

Note that
- You may train on different Math environments by simply changing the `env_id`
- `env_kwargs_json='{"use_mp": false}'` is needed only for Math environments

Then we show case another example of training LLMs on reasoning gym: 

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=rg:simple_equations groups_per_batch=64 group_size=8 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base
```