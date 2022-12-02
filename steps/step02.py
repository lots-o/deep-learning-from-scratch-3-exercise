"""
제 1 고지 : 미분 자동 계산 
    step 2 : 변수를 낳는 함수
        - Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현 
        - 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현 
    
"""
import numpy as np


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

    def forward(self, x):
        """
        구체적인 함수 계산 담당
        """
        raise NotImplementedError


class Square(Function):
    """
    y= x ** 2
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # NOTE : 0차원의 ndarray 의 경우 np.float64로 변환되는데(넘파이가 의도한 동작) 추후 step09 에서 처리
        return x**2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
