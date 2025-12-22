import torch.nn as nn


def get_policy_kwargs(net_arch_str="small", activation_fn=nn.Tanh):
  if net_arch_str is None:
    net_arch_str = "small"

  if activation_fn is None:
    activation_fn = nn.Tanh
  elif isinstance(activation_fn, str):
    if activation_fn.lower() == "tanh":
      activation_fn = nn.Tanh
    elif activation_fn.lower() == "relu":
      activation_fn = nn.ReLU
    elif activation_fn.lower() == "leakyrelu":
      activation_fn = nn.LeakyReLU
    else:
      raise ValueError(f"Unknown activation_fn string: {activation_fn} type: {type(activation_fn)}")

  if net_arch_str == "small":
    net_arch = dict(pi=[64, 64], vf=[64, 64], activation_fn=activation_fn)
  elif net_arch_str == "medium":
    net_arch = dict(pi=[128, 128], vf=[128, 128], activation_fn=activation_fn)
  elif net_arch_str == "large":
    net_arch = dict(pi=[256, 256, 256], vf=[256, 256, 256], activation_fn=activation_fn)
  else:
    raise ValueError(f"Unknown net_arch_str: {net_arch_str} type: {type(net_arch_str)}")

  return dict(net_arch=net_arch, activation_fn=activation_fn)
