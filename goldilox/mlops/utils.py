import os
import shutil

MLOPS = 'mlops'


def copy_file(base: str, path: str, filename: str, directory: str) -> bool:
    shutil.copyfile(str(base.parent.absolute().joinpath(MLOPS).joinpath(directory).joinpath(filename)),
                    os.path.join(path, filename))
    return True
