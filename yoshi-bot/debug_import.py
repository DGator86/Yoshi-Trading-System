import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import gnosis
print(f"gnosis location: {gnosis.__file__}")
import gnosis.utils
print(f"gnosis.utils location: {gnosis.utils.__file__}")
from gnosis.utils import drop_future_return_cols
print("Import successful")
