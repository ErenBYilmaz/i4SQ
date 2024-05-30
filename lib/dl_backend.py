import cachetools

from lib.backend_interface import DeepLearningBackendInterface


@cachetools.cached(cachetools.LRUCache(maxsize=1))
def b() -> DeepLearningBackendInterface:
    try:
        import tensorflow
        from lib.tf_backend_interface import TensorflowBackendInterface
        return TensorflowBackendInterface()
    except ImportError:
        from lib.torch_backend_interface import TorchBackendInterface
        return TorchBackendInterface()
