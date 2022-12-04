"""
제 1 고지 : 미분 자동 계산 
    step 2 : 변수를 낳는 함수
        - Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현 
        - 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현 
    
"""
import torch
import numpy as np
import torch.nn as nn


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class Function:
    """
    Function Base Class

    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        구체적인 함수 계산 담당
        # NOTE : 0차원의 ndarray 의 경우 np.float64로 변환되는데(넘파이가 의도한 동작) 추후 step09 에서 처리
        """
        raise NotImplementedError


class Square(Function):
    """
    y= x ^ 2
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2


class Sigmoid(Function):
    """
    y = 1 / (1 + e ^(-x))
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


# Dezero
x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

# Dezero ~ Pytorch
## Dezero
x = Variable(np.array(1))
f = Sigmoid()
y = f(x)
print(y.data)

## Pytorch
x = torch.Tensor([1])
f = nn.Sigmoid()
y = f(x)
print(y.data)
