import sys
import os

SAVE_DIR = sys.argv[1]
open(os.path.join(SAVE_DIR, 'kill'), 'w')