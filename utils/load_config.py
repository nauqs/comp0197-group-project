import configparser

def load_config(config_file="config.ini"):
    """Load configuration file
    
    Args:
        config_file (str): Path to configuration file.

    Returns:
        config (ConfigParser): Configuration file.
    """

    config = configparser.ConfigParser()
    config.read(config_file)
    
    return config