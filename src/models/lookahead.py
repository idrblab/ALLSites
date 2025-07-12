# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/8/20 20:47
@author: LiFan Chen
@Filename: lookahead.py
@Software: PyCharm
"""
from collections import defaultdict
import torch


class Lookahead:
    """Lookahead Optimizer Wrapper - 完全兼容版本"""
    
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        # 为每个参数组添加计数器
        for group in self.optimizer.param_groups:
            group["counter"] = 0
    
    @property
    def param_groups(self):
        """返回内部优化器的参数组"""
        return self.optimizer.param_groups
    
    @property  
    def defaults(self):
        """返回内部优化器的默认值"""
        return getattr(self.optimizer, 'defaults', {})
    
    @property
    def state_dict(self):
        """返回状态字典的属性版本"""
        return self.get_state_dict()
    
    def get_state_dict(self):
        """获取状态字典"""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def update(self, group):
        """更新慢权重"""
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        """更新所有参数组的慢权重"""
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        """执行一步优化"""
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def zero_grad(self, set_to_none=False):
        """清除梯度"""
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        
        # 重建 slow state
        for k, v in slow_state_dict["state"].items():
            self.state[k] = v
            
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        """添加参数组"""
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
        
    def __getattr__(self, name):
        """代理到内部优化器的其他属性"""
        if name in ['optimizer', 'k', 'alpha', 'state', 'fast_state']:
            return object.__getattribute__(self, name)
        return getattr(self.optimizer, name)