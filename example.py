from transformers import AutoTokenizer
import gem
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

env = gem.make("math:GSM8K")
tool = PythonCodeTool()
wrapped_env = ToolEnvWrapper(env, tools=[tool])
wrapped_env = WRAPPER_FACTORY["concat_chat"](
    wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
)
obs, info = wrapped_env.reset()

# we ignore the obs and use a dummy action
dummy_action = "<think>Let me compare 9.9 and 9.11 using python.</think><python>print('9.9 > 9.11?', 9.9 > 9.11)</python>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
# continue to sample the next response given the tool results ...

wrapped_env.close()