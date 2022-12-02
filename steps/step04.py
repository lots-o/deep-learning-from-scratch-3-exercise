"""
제 1 고지 : 미분 자동 계산 
    step 4 : 수치 미분
        - 미분 = 변화율 
        - 컴퓨터는 극한을 취할 수 없으니, 매우 작은 값 h 을 이용하여 함수의 변화량 계산 
        - forward difference 보단 centered difference 가 오차율이 적다 
        - 하지만 수치미분은 오차가 포함되어 있기 때문에 어떤 계산인지에 따라 오차가 커질 수 있다. 
        - 또한 변수가 여러개인 계산을 미분할 경우 변수 각각을 미분해야 하므로 계산량이 많다.
        - 그래서 등장한 것이 역전파 알고리즘 
        - 일반적으로 수치미분 ~ 역전파 결과를 비교하는 gradient checking을 통해 올바른 구현여부 확인 (step10)
        
        
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        구체적인 함수 계산 담당
        """
        raise NotImplementedError


class Square(Function):
    """
    y= x ** 2
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2


class Exp(Function):
    """
    y=e**x
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


def numerical_diff(f: Function, x: Variable, eps: float = 1e-4) -> np.ndarray:
    """
    calculate centered difference
    """
    x0 = Variable(x.data - eps)  # x - h
    x1 = Variable(x.data + eps)  # x + h
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)  # (f(x+h) - f(x-h)) / 2h


def f_composition(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f_composition, x)
print(dy)
