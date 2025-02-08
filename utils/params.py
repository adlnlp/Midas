import toml
import json

def get_params(config_file_path):
    config = json.loads(json.dumps(toml.load(config_file_path)))
    return config 