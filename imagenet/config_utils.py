import yaml
import json
from types import SimpleNamespace

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
        # if (k in dct and isinstance(dct[k], dict) ):    
            dict_merge(dct[k], merge_dct[k])
        else:
            if k in dct.keys():
                dct[k] = merge_dct[k]

class ConfigObj():
    def __init__(self, default_path, config_path=None) -> None:
        cfg = {}
        if config_path is not None:
            with open(config_path) as cf_file:
                cfg = yaml.load( cf_file.read(), Loader=yaml.Loader)     
        
        with open(default_path) as def_cf_file:
            default_cfg = yaml.load( def_cf_file.read(), Loader=yaml.Loader)

        dict_merge(default_cfg, cfg)
        self._data_obj = json.loads(json.dumps(default_cfg), object_hook=lambda item: SimpleNamespace(**item))
    def get(self):
        return self._data_obj
    def __str__(self):
        return str(self._data_obj)


if __name__ == "__main__":
    config = ConfigObj('config/test_def.yaml', 'config/test.yaml')
    data = config.get()
    print(data)