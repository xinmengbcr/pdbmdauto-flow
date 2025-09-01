import importlib


def import_all_from_submodule(module_name):
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        if hasattr(module, "__all__"):
            globals().update({name: getattr(module, name)
                              for name in module.__all__})
        else:
            names = [name for name in dir(module) if not name.startswith("_")]
            globals().update({name: getattr(module, name) for name in names})
    except ImportError as e:
        print(
            f"Warning: Some modules from {module_name} could not be imported. Error: {e}"
        )
