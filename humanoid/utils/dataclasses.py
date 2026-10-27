import ast
import re
from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any, Dict, TypeVar

from humanoid.utils.array import recursive_deepcopy


T = TypeVar("T")


def _recursive_asdict_dataclass(data: Any) -> Dict[str, Any]:
    def get_value(field):
        if field.default_factory is not MISSING:
            return field.default_factory()
        elif field.default is not MISSING:
            return field.default
        return MISSING

    result = {}
    for field in fields(data):
        value = getattr(data, field.name, get_value(field))
        if value is MISSING:
            # Skip fields with no default value
            continue
        result[field.name] = recursive_asdict(value)
    return result


def recursive_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return _recursive_asdict_dataclass(obj)
    elif isinstance(obj, dict):
        return {k: recursive_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_asdict(v) for v in obj)
    else:
        return obj


def default_field(obj):
    return field(default_factory=lambda: recursive_deepcopy(obj))


def class_factory(cls: T) -> T:
    return field(default_factory=cls)


def get_class(path: str) -> type:
    """
    Look up a class based on a dotpath.
    Fails if the path does not point to a class.

    >>> import my_module
    >>> from hydra.utils import get_class
    >>> assert get_class("my_module.MyClass") is my_module.MyClass
    """

    def _locate(path: str) -> Any:
        """
        Locate an object by name or dotted path, importing as necessary.
        This is similar to the pydoc function `locate`, except that it checks for
        the module from the given path from back to front.
        """
        if path == "":
            raise ImportError("Empty path")
        from importlib import import_module
        from types import ModuleType

        parts = [part for part in path.split(".")]
        for part in parts:
            if not len(part):
                raise ValueError(
                    f"Error loading '{path}': invalid dotstring."
                    + "\nRelative imports are not supported."
                )
        assert len(parts) > 0
        part0 = parts[0]
        try:
            obj = import_module(part0)
        except Exception as exc_import:
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_import)}"
                + f"\nAre you sure that module '{part0}' is installed?"
            ) from exc_import
        for m in range(1, len(parts)):
            part = parts[m]
            try:
                obj = getattr(obj, part)
            except AttributeError as exc_attr:
                parent_dotpath = ".".join(parts[:m])
                if isinstance(obj, ModuleType):
                    mod = ".".join(parts[: m + 1])
                    try:
                        obj = import_module(mod)
                        continue
                    except ModuleNotFoundError as exc_import:
                        raise ImportError(
                            f"Error loading '{path}':\n{repr(exc_import)}"
                            + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                        ) from exc_import
                    except Exception as exc_import:
                        raise ImportError(
                            f"Error loading '{path}':\n{repr(exc_import)}"
                        ) from exc_import
                raise ImportError(
                    f"Error loading '{path}':\n{repr(exc_attr)}"
                    + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
                ) from exc_attr
        return obj

    try:
        cls = _locate(path)
        if not isinstance(cls, type):
            raise ValueError(
                f"Located non-class of type '{type(cls).__name__}'"
                + f" while loading '{path}'"
            )
        return cls
    except Exception as e:
        raise e


TEMPLATE_PATTERN = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def resolve_templates(obj, context):
    """
    Recursively resolve "${...}" strings in all dataclass fields, including init=False fields.
    Works with nested dataclasses, dicts, and lists. Mutates in place.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = _resolve_string(v, context)
            else:
                resolve_templates(v, context)
    # elif isinstance(obj, list):
    #     for i in range(len(obj)):
    #         v = obj[i]
    #         if isinstance(v, str):
    #             obj[i] = _resolve_string(v, context)
    #         else:
    #             resolve_templates(v, context)
    if is_dataclass(obj):
        for f in fields(obj):
            try:
                v = getattr(obj, f.name)
            except AttributeError:
                continue  # field(init=False) may not be set yet

            if v is None:
                continue

            if isinstance(v, str):
                try:
                    resolved = _resolve_string(v, context)
                    setattr(obj, f.name, resolved)
                except Exception:
                    pass  # You may choose to log this
            else:
                resolve_templates(v, context)


def _resolve_string(s: str, context: Any) -> Any:
    # Replace all ${...} with resolved values (repr ensures safe Python syntax)
    def replacer(match):
        path = match.group(1)
        value = _get_attr_from_path(context, path)
        return repr(value)  # repr adds quotes for strings, keeps numbers/bools safe

    resolved_str = TEMPLATE_PATTERN.sub(replacer, s.strip())

    # Try to evaluate the full expression if it's not a simple literal
    try:
        return ast.literal_eval(resolved_str)
    except Exception:
        pass

    try:
        # Evaluate full expression in restricted scope
        return eval(resolved_str, {"__builtins__": {}}, {})
    except Exception:
        # Fallback to the string itself if evaluation fails
        return resolved_str


def _get_attr_from_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj
