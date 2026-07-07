import sys
import types
import warnings


def using_mmcv_lite():
    try:
        import importlib.metadata as metadata
    except ImportError:
        return False
    try:
        metadata.version("mmcv-lite")
        return True
    except metadata.PackageNotFoundError:
        return False


def patch_mmcv_ops_for_lite():
    warnings.filterwarnings("ignore", message="Fail to import ``MultiScaleDeformableAttention``.*")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    if not using_mmcv_lite():
        return
    if "mmcv.ops" in sys.modules:
        return
    ops = types.ModuleType("mmcv.ops")
    ops.__file__ = "mmcv_ops_stub.py"
    ops.__path__ = []

    def get_attr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

    ops.__getattr__ = get_attr
    sys.modules["mmcv.ops"] = ops


def patch_torch_load_for_openmmlab():
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.load, "_ntu_rtmw_patched", False):
        return
    original_load = torch.load

    def load_with_openmmlab_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_openmmlab_default._ntu_rtmw_patched = True
    torch.load = load_with_openmmlab_default


def patch_runtime():
    patch_mmcv_ops_for_lite()
    patch_torch_load_for_openmmlab()
