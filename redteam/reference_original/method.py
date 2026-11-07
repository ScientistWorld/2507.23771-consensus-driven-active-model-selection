from pathlib import Path
import shutil
def generate(out, folder):
    # Audit-only restoration of original trained artifacts, not a candidate shortcut.
    source = Path('/home/user/redteam/_state/original_active')
    for task in ('glue_rte', 'glue_sst2'):
        shutil.copytree(source / task, out / task)
