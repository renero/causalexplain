DEFAULT_ITERATIVE_TRIALS: int = 20
DEFAULT_HPO_TRIALS: int = 20
DEFAULT_BOOTSTRAP_TRIALS: int = 20
DEFAULT_BOOTSTRAP_TOLERANCE: float = 0.3
DEFAULT_BOOTSTRAP_SAMPLING_SPLIT: float = 0.2
DEFAULT_SEED: int = 42
DEFAULT_MAX_CSV_LINES: int = 2000
DEFAULT_MAX_SAMPLES: int = 512

HEADER_ASCII = """   ___                      _                 _       _
  / __\\__ _ _   _ ___  __ _| | _____  ___ __ | | __ _(_)_ __
 / /  / _` | | | / __|/ _` | |/ _ \\ \\/ / '_ \\| |/ _` | | '_ \\
/ /__| (_| | |_| \\__ \\ (_| | |  __/>  <| |_) | | (_| | | | | |
\\____/\\__,_|\\__,_|___/\\__,_|_|\\___/_/\\_\\ .__/|_|\\__,_|_|_| |_|
                                       |_|"""

SUPPORTED_METHODS = ['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears']
DEFAULT_REGRESSORS = ['nn', 'gbt']
