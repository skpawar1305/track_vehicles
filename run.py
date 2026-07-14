import sys
sys.path = [p for p in sys.path
            if 'robostack' not in p and 'tyno_ws' not in p and 'python3.12' not in p]
from main import main
main()
