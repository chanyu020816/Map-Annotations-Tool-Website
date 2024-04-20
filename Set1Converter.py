import os
import argparse
from tqdm import tqdm

def set1convert(file, output_folder):
    with open(file, 'r') as f:
        lines = f.readlines()

    converted_line = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        class_id = int(part[0]) - 1
        rest = ' '.join(part[1:])
        converted_line = f'{class_id} {rest}\n'
        converted_lines.append(converted_line)

    with open(os.path.join(output_folder, file), 'w') as f:
        f.writelines(converted_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=int, default=1)
    parser.add_argument('--input-folder', type=Path, default='./',
        help='files need to be converted')
    parse.add_argument('--output-folder', type=Path, default='./',
        help='folder where converted file to be saved')

    opt = parse.parse_args()

    if opt.set == 1:
        for file in tqdm(os.listdir(opt.input_folder)):
            set1convert(file)
    else:
        pass    


