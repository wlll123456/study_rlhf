# %% [markdown]
# # 代码实现ppo

# %% [markdown]
# trl代码中的对于ppo的实现
# https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

# %% [markdown]
# 下面为你解释这些参数的含义：
# 
# ### 模型架构相关参数
# 1. **`vocab_size = 10`**
# 词汇表的大小代表了模型能够识别的不同词汇的数量。举例来说，若你正在处理的是一个简单的数字文本任务，其中仅有 0 - 9 这 10 个数字，那么 `vocab_size` 就会被设定为 10。
# 
# 2. **`hidden_size = 128`**
# 隐藏层的维度大小表明了模型中每个隐藏层神经元的数量。在神经网络里，隐藏层会对输入数据进行特征提取与转换。`hidden_size` 越大，模型所能学习到的特征就越复杂，不过这也会使计算量和内存需求增加。
# 
# 3. **`intermediate_size = 256`**
# 在 Transformer 架构里，`intermediate_size` 指的是前馈神经网络（FFN）中间层的维度。FFN 一般由两个线性层构成，中间层的维度通常会比输入输出层的维度大，这样有助于模型学习到更丰富的特征。
# 
# 4. **`num_hidden_layers = 2`**
# 隐藏层的数量意味着模型中堆叠的隐藏层的层数。层数越多，模型的表达能力就越强，能够学习到更复杂的模式，但同时也会增加过拟合的风险以及训练的难度。
# 
# 5. **`num_attention_heads = 4`**
# 注意力头的数量是指在多头注意力机制中并行的注意力头的个数。多头注意力机制能够让模型从不同的表示子空间中捕捉特征，提升模型的表达能力。
# 
# 6. **`num_key_value_heads = 4`**
# 键值对注意力头的数量在某些改进的注意力机制中会用到，它决定了用于计算键（key）和值（value）的注意力头的数量。在标准的多头注意力机制里，`num_key_value_heads` 通常和 `num_attention_heads` 相等。
# 
# ### 数据处理和生成相关参数
# 7. **`batch_size = 5`**
# 批量大小代表了在一次训练或者推理过程中同时处理的样本数量。使用较大的批量大小能够提升训练效率，但会增加内存的需求；而较小的批量大小则可以减少内存使用，但会使训练速度变慢。
# 
# 8. **`length_x = 5`**
# 输入序列的长度指的是每个输入样本的长度。在处理文本时，它代表的是输入文本中词元（token）的数量。
# 
# 9. **`max_new_tokens = 5`**
# 最大新生成的词元数量表示在文本生成任务中，模型最多可以生成的词元数量。例如在文本续写任务里，这个参数会限制模型生成的文本长度。 

# %%
vocab_size = 10   #当前教程实际使用的时候是词汇表实际大小
hidden_size = 128
intermediate_size = 256
num_hidden_layers = 2
num_attention_heads = 4
batch_size = 3
length_x = 5
max_new_tokens = 5

# %% [markdown]
# ## 初始化actor模型
# 
# 以GPT2为例，初始化模型

# %%
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(1)

# 定义参数
vocab_size = 10
hidden_size = 128
intermediate_size = 256
num_hidden_layers = 2
num_attention_heads = 4

# 加载模型配置
config = GPT2Config(
    vocab_size=50257,
    n_embd=hidden_size,
    n_inner=intermediate_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads
)

# 初始化 GPT - 2 模型
model = GPT2LMHeadModel(config)

# %% [markdown]
# ## model generate
# 
# 主要看下inputs_ids和attention_mask的含义

# %% [markdown]
# ### inputs_ids
# 
# input_ids：它是一个张量（tensor），表示文本被分词后每个词（token）对应的 ID。比如在第一行 [20015, 232, 25465, ...] 中，每个数字都是原文本中一个词被 GPT - 2 分词器转换后的唯一标识。不同模型的词表不同，这些 ID 对应的具体词汇也不一样。这里第一行可能对应一句中文文本分词结果，第二行 [14150, 257, 922, ...] 前半部分对应英文文本，后半部分 50256 一般是填充值 ，表示补齐固定长度。
# 
# 
# attention_mask：同样是张量，用于指示哪些位置是有效的词（值为 1），哪些位置是填充的（值为 0） 。比如第二行 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] 表示前 4 个词是有效输入，后面是填充的，模型在处理时会忽略填充位置。

# %% [markdown]
# inputs_ids可以认为是要输入的文本经过tokenizer处理后的结果，而attention_mask则是用于指示哪些位置是有效的词（值为 1），哪些位置是填充的（值为 0） 。

# %%
from transformers import GPT2Tokenizer
import torch

# 初始化 GPT - 2 分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# 设置padding token
tokenizer.pad_token = tokenizer.eos_token  # 使用EOS token作为padding token

# 输入文本
inputs = ['今天天气不错', 'have a good day']

# 对输入进行分词处理
inputs = tokenizer(inputs, return_tensors='pt',padding=True, truncation=True)

print(inputs)

# %%
output_ids = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens)

print(output_ids)


# %%
output_ids = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(output_ids)

# %% [markdown]
# 填充左边和右边会导致input_ids中padding_id的位置不一样，导致attention_mask中padding_id的位置不一样，导致模型在处理时会忽略填充位置。

# %%
tokenizer.padding_side = 'left'
inputs = ['今天天气不错', 'have a good day']
inputs = tokenizer(inputs, return_tensors='pt',padding=True, truncation=True)

print(inputs)

output_ids = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens)

print(output_ids)

output_ids = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(output_ids)

# %% [markdown]
# ## 初始化reward model

# %% [markdown]
# 根据之前的定义，奖励模型可以从模型的输出中提取出最后一个token的隐藏状态，然后通过一个线性层计算奖励。

# %% [markdown]
# 假设batch_size = 2, sequence_length = 4
# input_ids = torch.tensor([
#     [1, 2, 3, 4],  # 第一个序列
#     [5, 6, 7, 8]   # 第二个序列
# ])
# 
# attention_mask = torch.tensor([
#     [1, 1, 1, 0],  # 第一个序列有效长度为3
#     [1, 1, 1, 1]   # 第二个序列有效长度为4
# ])
# 
# sequence_length = attention_mask.sum(dim=1).long() - 1
# 
# 结果: tensor([2, 3])
# 
# 第一个序列：3-1=2（索引从0开始）
# 
# 第二个序列：4-1=3
# 
# batch_indices = torch.arange(batch_size)
# 
# 结果: tensor([0, 1])
# 
# 假设hidden_size = 2
# 
# last_hidden_state = torch.tensor([
#     [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]],  # 第一个序列
#     [[5.0, 5.1], [6.0, 6.1], [7.0, 7.1], [8.0, 8.1]]   # 第二个序列
# ])
# 
# 使用batch_indices和sequence_length提取
# 
# result = last_hidden_state[batch_indices, sequence_length]
# 
# 结果: tensor([[3.0, 3.1],    # 第一个序列的第2个位置（索引从0开始）
# 
# [8.0, 8.1]])   # 第二个序列的第3个位置

# %%
class GPTRewardModel(torch.nn.Module):
    def __init__(self, gpt_model, reward_head):
        super(GPTRewardModel, self).__init__()
        self.gpt_model = gpt_model
        self.reward_head = reward_head
        
    def forward(self, input_ids, attention_mask):
        # 获取模型的输出
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        # 通常取最后一个隐藏状态作为输出
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        # 确保sequence_length是long类型
        sequence_length = attention_mask.sum(dim=1).long() - 1
        
        # 使用torch.arange并确保在正确的设备上
        batch_indices = torch.arange(batch_size, device=input_ids.device).long()
        last_hidden_state = last_hidden_state[batch_indices, sequence_length]
        
        # 计算奖励
        rewards = self.reward_head(last_hidden_state)
        return rewards

# 重新初始化模型
model.config.output_hidden_states = True
rm_model = GPTRewardModel(model, torch.nn.Linear(hidden_size, 1))

# %%
reward = rm_model(inputs['input_ids'], inputs['attention_mask'])
print(reward)

# %% [markdown]
# ## 简化版ppo
# 从以上过程可以看出，我们输入给模型的其实是input_ids和attention_mask，所以我们现在为了展示方便，构造一个没有实际意义的输入，输入给模型，然后输出奖励。

# %%
prompt = torch.randint(0, vocab_size, (batch_size, length_x))
response = torch.randint(0, vocab_size, (batch_size, length_x + max_new_tokens))

# %%
print(prompt)
print(response)

# %% [markdown]
# 我们希望让模型只关注response，所以对prompt对应的mask置为0

# %%
attention_mask = torch.ones(batch_size, length_x+max_new_tokens)
attention_mask[:, :length_x] = 0
print(attention_mask)


# %%
prompt_attention_mask = torch.ones(batch_size, length_x)
prompt_attention_mask

# %% [markdown]
# 创建几个模型
# 
# 
# model_ref 和model的配置一样
# 
# reward model和value model的配置大体一样
# 
# value model的输出是所有token的隐藏状态所得到的value

# %%
# 初始化 GPT - 2 模型
model_ref = GPT2LMHeadModel(config)

# %% [markdown]
# 查看区别

# %%
print(model_ref)
print(model)

# %% [markdown]
# ## 初始化value model

# %% [markdown]
# 假设我们有以下维度的数据：
# 
# last_hidden_state 的形状是 [batch_size, sequence_length, hidden_size]
# 
# 比如 [5, 10, 128]，表示批次大小为5，序列长度为10，隐藏层维度为128
# 
# self.value_head 是一个线性层 Linear(hidden_size, 1)
# 
# 输入维度是128，输出维度是1
# 
# 处理过程：
# 
# self.value_head(last_hidden_state) 的操作：
# 
# 输入: [5, 10, 128]
# 
# 输出: [5, 10, 1] # 线性层将最后一个维度从128转换为1
# 
# [:, :, 0] 的操作：
# 
# 取最后一个维度的第0个元素
# 
# 结果形状变为: [5, 10]

# %%
class GPTValueModel(torch.nn.Module):
    def __init__(self, gpt_model, value_head):
        super().__init__()
        self.gpt_model = gpt_model
        self.value_head = value_head
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        
        values = self.value_head(last_hidden_state)[:, :, 0]

        return values
    
model.config.output_hidden_states = True
vm_model = GPTValueModel(model,torch.nn.Linear(hidden_size, 1))

# %%
print(rm_model)
print(vm_model)

# %% [markdown]
# ## ppo前向过程

# %% [markdown]
# 创建几个model的函数

# %%
def get_response(model, prompt, max_new_tokens):
    inputs = {'input_ids': prompt}  # ignore mask，好像不需要mask
    y = model.generate(**inputs,
                       max_new_tokens=max_new_tokens,
                       # forced_eos_token_id=True
                       )
    return y

def get_reward(model, response, attention_mask):
    inputs   = {'input_ids': response, 'attention_mask': attention_mask}  # ignore mask
    y = model(inputs['input_ids'], inputs['attention_mask'])
    return y

def get_value(model, prompt, attention_mask):
    inputs = {'input_ids': prompt, 'attention_mask': attention_mask}  # ignore mask
    y = model(inputs['input_ids'], inputs['attention_mask'])
    return y

# %%
prompt

# %%
response

# %%
prompt_attention_mask

# %%
attention_mask

# %%
print(get_response(model, prompt, max_new_tokens, prompt_attention_mask))
print(get_reward(rm_model, response, attention_mask))
print(get_value(vm_model, response, attention_mask))


# %% [markdown]
# PPO 相关设置

# %% [markdown]
# 封装几个ppo的model

# %%
class PPOModels():
    def __init__(self, model_actor, model_ref, model_rm, model_critic):
        self.actor = model_actor
        self.ref = model_ref
        self.rm = model_rm
        self.critic = model_critic


model_ref.eval()
rm_model.eval()
models = PPOModels(model, model_ref, rm_model, vm_model)


# %% [markdown]
# 设置ppo的超参数

# %% [markdown]
# 1. ppo_epochs在每次策略更新时，PPO 算法对收集到的数据进行迭代训练的次数。
# 
# 2. mini_batch_size每个训练步骤中，从收集到的数据里选取的小批量数据的样本数量。
# 
# 3. epochs整个训练过程中，算法对所有收集到的数据进行完整遍历的次数。
# 
# 4. kl_ctlKL 散度惩罚项的系数，用于控制新旧策略之间的差异程度。
# 
# 5. vf_coef价值函数损失的系数，用于平衡策略损失和价值函数损失在总损失中的权重。
# 
# 6. lam广义优势估计（GAE）中的 \(\lambda\) 参数，用于平衡优势估计的偏差和方差。
# 
# 7. gamma折扣因子，用于计算未来奖励的折现值，决定未来奖励在当前价值估计中的重要程度。
# 
# 8. cliprange_value价值函数裁剪范围的参数，用于限制价值函数更新的幅度

# %%
class PPOConfig():
    def __init__(self):
        self.ppo_epochs = 5
        self.mini_batch_size = 2
        self.epochs = 4
        self.kl_ctl = 0.1
        self.vf_coef = 0.1
        self.lam = 0.9
        self.gamma = 0.9
        self.cliprange_value = 0.2

    def __str__(self):
        return f'ppo_epochs:{self.ppo_epochs}\nmini_batch_size:{self.mini_batch_size}\nepochs:{self.epochs}\nkl_ctl:{self.kl_ctl}'


ppo_config = PPOConfig()

# %% [markdown]
# 在每一步中ppo都在干什么

# %% [markdown]
# 首先要有个列表来记录每一步的采样

# %%
ppo_old_batchs = {
    'prompt': None,
    'response': None,
    'mask': None,
    'logprobs_ref': None,
    'logprobs_old': None,
    'logprobs': None,
    'values_old': None,
    'values': None,
    'rewards': None,
    'rewards_kl': None,
    'loss': None,
    'logits': None,
}

ppo_old_batchs['prompt'] = prompt
ppo_old_batchs['response'] = response
ppo_old_batchs['mask'] = attention_mask

# %%
ppo_old_batchs

# %% [markdown]
# 前向推理，得到token的logprobs

# %% [markdown]
# logprobs = F.log_softmax(logits, dim=-1)第一步:对logits进行softmax并取log
# 
# torch.gather是一个用于从张量中按索引收集值的操作 
# 
# 假设我们有:
# 
# logp.shape = [1, 5, 32]      # [batch_size, seq_len, vocab_size]
# 
# labels.shape = [1, 5]        # [batch_size, seq_len]
# 
# 1. labels.unsqueeze(2)
# 
# 在最后增加一个维度
# 
# labels_expanded = labels.unsqueeze(2)   # shape变为[1, 5, 1]
# 
# 2. torch.gather(logp, 2, labels_expanded)
# 
# dim=2表示在词表维度(第3维)上收集值
# 
# gathered = torch.gather(logp, 2, labels_expanded)  # shape为[1, 5, 1]
# 
# 3. squeeze(-1)
# 
# 去掉最后一个维度
# 
# logpy = gathered.squeeze(-1)  # 最终shape为[1, 5]

# %%
import torch.nn.functional as F

def get_logits(model, input_ids):
    # 得到logits
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    return logits

def get_logprobs(model, response, attention_mask):
    # 得到logprobs
    logits = get_logits(model, response)
    # F.log_softmax() 是先进行softmax运算然后再取对数（log）
    all_token_logprobs = F.log_softmax(logits, dim=-1)
    # 使用torch.gather() 从logprobs中收集response的值
    gathered = torch.gather(all_token_logprobs, 2, response.unsqueeze(2))
    # 去掉最后一个维度
    response_logprobs = gathered.squeeze(-1)
    return response_logprobs

logprobs_ref = get_logprobs(models.ref, ppo_old_batchs['response'], ppo_old_batchs['mask'])
logprobs_old = get_logprobs(models.actor, ppo_old_batchs['response'], ppo_old_batchs['mask'])
logprobs = get_logprobs(models.actor, ppo_old_batchs['response'], ppo_old_batchs['mask'])

print(logprobs_ref.shape)
print(logprobs_old.shape)
print(logprobs.shape)   


# %%
response.shape

# %%
logprobs

# %% [markdown]
# 计算kl

# %%
def get_kl(logprobs_ref, logprobs_old, kl_ctl):
    kl = logprobs_ref - logprobs_old
    kl = kl * kl_ctl
    return kl

kl = get_kl(logprobs_ref, logprobs_old, ppo_config.kl_ctl)
print(kl)


# %% [markdown]
# 计算reward_kl
# 

# %%
def get_reward_with_kl(logprobs_ref, logprobs_old, kl_ctl, reward):
    kl = logprobs_ref - logprobs_old
    kl = kl * kl_ctl
    kl[:, -1] += reward[:, 0]
    return kl

print(kl)
rewards = get_reward(models.rm, ppo_old_batchs['response'], ppo_old_batchs['mask'])
print(rewards)

kl_reward = get_reward_with_kl(logprobs_ref, logprobs_old, ppo_config.kl_ctl, rewards)
print(kl_reward)


# %%
values = get_value(models.critic, ppo_old_batchs['response'], ppo_old_batchs['mask'])

# %%
values

# %%
ppo_old_batchs['logprobs_ref'] = logprobs_ref
ppo_old_batchs['logprobs_old'] = logprobs_old
ppo_old_batchs['logprobs'] = logprobs
ppo_old_batchs['values_old'] = values
ppo_old_batchs['rewards'] = rewards
ppo_old_batchs['rewards_kl'] = kl_reward

ppo_old_batchs

# %% [markdown]
# ## 计算loss

# %% [markdown]
# rewards：一个张量，代表在每个时间步获得的奖励。
# 
# mask：一个掩码张量，用于标识哪些时间步是有效的（例如，用于处理终止状态）。
# 
# values：一个张量，代表每个时间步的状态价值估计。
# 
# gamma：折扣因子，用于计算未来奖励的折现值，取值范围通常在 [0, 1] 之间。
# 
# lam：GAE 中的 \(\lambda\) 参数，用于平衡偏差和方差，取值范围同样在 [0, 1] 之间。

# %%
def get_GAE(rewards, attention_mask, values, gemma, lam):
    lastgae = 0 #初始化为 0，用于存储上一个时间步的广义优势估计值。
    advantages_recersed = []
    response_len = rewards.shape[-1]

    values = values * attention_mask
    rewards = rewards * attention_mask

    for t in reversed(range(response_len)):
        nextvalues = values[:, t + 1] if t < response_len - 1 else 0.0
        # 计算时间步 t 的 TD 误差（Temporal Difference error），即当前奖励加上折扣后的下一个时间步的价值估计，再减去当前时间步的价值估计。
        delta = rewards[:, t] + gemma * nextvalues - values[:, t]
        # 根据 GAE 的递推公式，计算当前时间步的广义优势估计值。
        lastgae = delta + gemma * lam * lastgae
        advantages_recersed.append(lastgae)
    # 将 advantages_reversed 列表反转，使其按时间步的正序排列。
    advantages = torch.stack(advantages_recersed[::-1]).transpose(0, 1)
    return advantages


# %%
ppo_old_batchs

# %%
gae = get_GAE(ppo_old_batchs['rewards'], ppo_old_batchs['mask'], ppo_old_batchs['values_old'], ppo_config.gamma, ppo_config.lam)
gae

# %%
gae = get_GAE(ppo_old_batchs['rewards_kl'], ppo_old_batchs['mask'], ppo_old_batchs['values_old'], ppo_config.gamma, ppo_config.lam)
gae


# %% [markdown]
# 计算value loss
# 

# %% [markdown]
# advantages：优势函数的估计值，用于计算回报。
# 
# 
# values：当前价值函数的估计值。
# 
# values_old：旧的价值函数估计值。
# 
# mask：掩码张量，用于指定哪些元素参与损失计算。
# 
# cliprange_value：裁剪范围，用于限制价值函数的更新幅度。

# %% [markdown]
# https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/ppo_trainer.py#L561C29-L567C30

# %%
def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def get_value_loss(advantages, values, values_old, attention_mask, cliprange_value):
    returns = values_old + advantages
    advantages = advantages.detach()

    vpredclipped = torch.clamp(values, values_old - cliprange_value, values_old + cliprange_value)

    vf_losses1 = torch.square(vpredclipped - returns)
    vf_losses2 = torch.square(values - returns)
    vf_loss_max = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * masked_mean(vf_loss_max, attention_mask)
    return vf_loss



# %%
ppo_old_batchs['values'] = ppo_old_batchs['values_old'] + 0.5

# %%
value_loss = get_value_loss(gae, ppo_old_batchs['values'], ppo_old_batchs['values_old'], ppo_old_batchs['mask'], ppo_config.cliprange_value)
value_loss

# %% [markdown]
# 计算policy loss
# https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/ppo_trainer.py#L569-L574

# %%
def get_policy_loss(advantages, logprobs, logprobs_old, mask, cliprange):
    # 重要性采样
    ratio = torch.exp(logprobs - logprobs_old)
    # 计算策略损失
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss_max = torch.max(pg_losses, pg_losses2)
    pg_loss = masked_mean(pg_loss_max, mask)
    return pg_loss



# %%
pg_loss = get_policy_loss(gae, ppo_old_batchs['logprobs'], ppo_old_batchs['logprobs_old'], ppo_old_batchs['mask'], ppo_config.cliprange_value)

# %%
pg_loss

# %% [markdown]
# 计算熵损失
# https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/ppo_trainer.py#L582-L583

# %% [markdown]
# entropy（熵）没有直接参与到模型的损失（loss）
# 
# 在计算完损失并进行反向传播和参数更新后，代码计算了 entropy
# 
# 这里计算的 entropy 被记录到 entropy_stats 张量中，用于后续的统计和记录，但没有用于损失计算。

# %%
logits = get_logits(models.actor, ppo_old_batchs['response'])
ppo_old_batchs['logits'] = logits

# %%
def get_entropy_loss(logits, mask):
    prob_dist = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
    return entropy

entropy = get_entropy_loss(ppo_old_batchs['logits'], ppo_old_batchs['mask'])
entropy
                                

# %%
loss = pg_loss + ppo_config.vf_coef * value_loss

# %%
def get_loss(batchs, ppo_config):
    gae = get_GAE(batchs['rewards_kl'],
                  batchs['mask'],
                  batchs['values'],
                  ppo_config.gamma,
                  ppo_config.lam)
    value_loss = get_value_loss(gae,
                             batchs['values'],
                             batchs['values_old'],
                             batchs['mask'],
                             ppo_config.cliprange_value)
    pg_loss = get_policy_loss(
                              gae,
                              batchs['logprobs'],
                              batchs['logprobs_old'],
                              batchs['mask'],
                              ppo_config.cliprange_value)
    entropy = get_entropy_loss(batchs['logits'], batchs['mask'])
    loss = pg_loss + ppo_config.vf_coef * value_loss
    return loss

# %%
loss = get_loss(ppo_old_batchs, ppo_config)
loss

# %%
ppo_old_batchs

# %% [markdown]
# ## PPO训练
# 
# https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/ppo_trainer.py#L529-L538

# %% [markdown]
# 将一个完整的批次数据 ppo_batchs 按照指定的 batch_size 和 mini_batch_size 划分成多个小批次数据

# %%
import numpy as np
def get_minibatch(ppo_batchs, batch_size, mini_batch_size):
    # 计算需要多少个小批次
    step = batch_size // mini_batch_size
    ppo_batchs_iter = []
    
    # 随机打乱索引以提高训练效果
    b_inds = np.random.permutation(batch_size)
    
    # 根据索引创建小批次
    for i in range(step):
        start_idx = i * mini_batch_size
        end_idx = start_idx + mini_batch_size
        batch_inds = b_inds[start_idx:end_idx]
        
        # 创建当前小批次的数据
        mini_batch = {}
        for key, value in ppo_batchs.items():
            if value is not None and isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                mini_batch[key] = value[batch_inds]
            else:
                mini_batch[key] = value
                
        ppo_batchs_iter.append(mini_batch)
    
    return ppo_batchs_iter

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# %%
ppo_old_batchs

# %%
def ppo_train_step(models, ppo_batchs, ppo_config, get_loss, optimizer):
    losses = []
    
    
    # 多轮PPO训练
    for i in range(ppo_config.ppo_epochs):
        # 获取小批次数据
        ppo_batchs_iter = get_minibatch(
            ppo_batchs, batch_size, ppo_config.mini_batch_size)
        
        # 对每个小批次进行训练
        for mini_batchs in ppo_batchs_iter:
            # 获取当前策略的输出
            optimizer.zero_grad()
            # 重新计算所有中间结果，而不是重用之前的计算图
            with torch.set_grad_enabled(True):
                logits = get_logits(models.actor, mini_batchs['prompt'])
                """
                省略了
                """

                
                # 计算损失
                loss= get_loss(
                    mini_batchs, ppo_config)
                
                # 在实际训练中应该进行反向传播
                loss.backward()
            optimizer.step()
            
            # 记录损失
            losses.append(loss)
    
    # 更新批次数据中的损失
    ppo_batchs['loss'] = losses
    
    print(losses)




