import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCENARIOS = Path(__file__).with_name("extension_scenarios.js")
NAMES = re.findall(r'^  async "(.+?)"\(\)', SCENARIOS.read_text(), re.MULTILINE)


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
@pytest.mark.parametrize("name", NAMES)
def test_extension(name):
    subprocess.run(["node", str(SCENARIOS), name], check=True, timeout=10)
