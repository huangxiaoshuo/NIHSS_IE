import sys
sys.path.append('..')
from pathlib import Path
BASE_DIR = Path('./main')
config = {
    'data_dir': BASE_DIR / 'dataset',
    'ann_data': BASE_DIR / 'ann_data',
    'log_dir': BASE_DIR / 'output/log',
    'checkpoint_dir': BASE_DIR / 'output/checkpoints',
    'result_dir' : BASE_DIR / 'output/result',
    'figure_dir' : BASE_DIR / 'output/figure'
}