import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/andres/repositories/Tree-Transformer')
from solver import Solver


class SentenceGenerator(Solver):
    def __init__(self, args):
        super(SentenceGenerator, self).__init__(args)

