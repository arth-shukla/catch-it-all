from functools import partial
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", construct_python_tuple
)

OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
OmegaConf.to_container = partial(OmegaConf.to_container, resolve=True)


def parse_cfg(
    cfg_path: str = None, default_cfg_path: str = None, use_cli=True
) -> OmegaConf:
    if default_cfg_path is not None:
        base = OmegaConf.load(default_cfg_path)

        if "base_config" in base:
            old_cfg = None
            base_config = base.base_config
            if isinstance(base_config, str):
                base_config = [base_config]
            if not isinstance(base_config, list):
                raise ValueError("base_config must be a string or list of strings")
            for path in base_config:
                new_path = Path(default_cfg_path).parent / Path(path)
                new_cfg = parse_cfg(default_cfg_path=new_path)
                if old_cfg is not None:
                    new_cfg = OmegaConf.merge(old_cfg, new_cfg)
                old_cfg = new_cfg

            base = OmegaConf.merge(old_cfg, base)
    else:
        base = OmegaConf.create()

    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
        base.merge_with(cfg)

    if use_cli:
        cli = OmegaConf.from_cli()
        for k, v in cli.items():
            if v is None:
                cli[k] = True
        base.merge_with(cli)
    return base
