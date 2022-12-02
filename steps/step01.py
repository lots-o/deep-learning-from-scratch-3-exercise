"""
제 1 고지 : 미분 자동 계산 
    step 1 : 상자로서의 변수
    - 상자와 데이터는 별개다
    - 상자에는 데이터가 들어간다(대입 혹은 할당)
    - 상자 속을 들여다보면 데이터를 알 수 있다.(참조)
"""
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)
