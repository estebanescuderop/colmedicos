_REGISTRY = {}

def register(name: str):
    def deco(func):
        _REGISTRY[name] = func
        return func
    return deco

def call(name: str, *args, **kwargs):
        if name not in _REGISTRY:
            raise KeyError(f"Función no registrada: {name}")
        return _REGISTRY[name](*args, **kwargs)
