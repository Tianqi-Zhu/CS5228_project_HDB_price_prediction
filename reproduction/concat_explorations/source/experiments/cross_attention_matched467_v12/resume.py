"""Resume only after verifying guarded inputs and backend."""
import sys
from experiments.cross_attention_matched467_v12.verify import check_inputs
from experiments.cross_attention_matched467_v12 import run
if __name__=='__main__':
    check_inputs(); sys.argv=[sys.argv[0],'--resume']; run.main()
