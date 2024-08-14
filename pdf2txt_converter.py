import pathlib
import subprocess
import multiprocessing

# путь к директории с pdf-файлами
dir_path_pdf = r'f:\\projects\\tilllama_win\\data\\fincert\\pdf\\'

# путь к директории с результами распознавания txt
dir_path_txt = r'f:\\projects\\tilllama_win\\data\\fincert\\txt\\'

path_work = pathlib.Path(dir_path_pdf)

path_to_frcmd = 'c:\\Program Files (x86)\\ABBYY FineReader 15\\FineCmd.exe'

for k, entry in enumerate(path_work.iterdir()):
    if entry.is_file():
        path_to_save = pathlib.Path(dir_path_txt, entry.name).with_suffix('.txt')
        result = subprocess.run([path_to_frcmd,
                                 entry,
                                 '/lang', 'Russian', 'English',
                                 '/out', path_to_save])
        print(f"{k}:{entry.name} = {result.returncode}")
