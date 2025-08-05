import yaml
from pathlib import Path

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Write configuration to YAML file.")
    parser.add_argument('--config-path', type=Path, help='Path to the configuration file.')
    
    args = parser.parse_args()

    # read the configuration from the specified file
    with open(args.config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['project_path'] =  str(args.config_path.parent)

    # Write the configuration to the specified file
    with open(args.config_path.with_suffix('.updated.yaml'), 'w') as file:
        yaml.safe_dump(config_dict, file)
