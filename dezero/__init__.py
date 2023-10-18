# =============================================================================
# step23.py부터 step32.py까지는 simple_core를 이용
is_simple_core = False
# step33 부터는  dezero/core.py 로 대체한다.

# =============================================================================

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
    from dezero.core_simple import Config

else:
    # step33 부터 dezero/core.py 정의
    from dezero.core import Variable
    from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Config
    from dezero.layers import Layer
    from dezero.models import Model

    import dezero.functions
    import dezero.utils
    import dezero.layers
    import dezero.optimizers
    import dezero.datasets

setup_variable()
__version__ = "0.0.13"
