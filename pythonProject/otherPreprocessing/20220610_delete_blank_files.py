import os

def main(path):
    paper_number_for_training = 0
    for curDir, dirs, files in os.walk(path):
        files = [file for file in files if (not file.startswith('.'))]
        if files:
            print(f'\n\n now at {curDir}')
            all = len(files)
            file_number = 0
            file_deleted = 0
            for file in files:
                file_path = os.path.join(curDir, file)
                with open(file_path, mode='r', encoding='utf-8') as f:
                    content = f.read()
                    if content == '':
                        os.remove(file_path)
                        file_deleted += 1
                        print(f'\r file deleted: {file_deleted}, file counted: {file_number}, all: {all}', end='')
                    else:
                        file_number += 1
                        paper_number_for_training += 1
                        print(f'\r file deleted: {file_deleted}, file counted: {file_number}, all: {all}', end='')

    print(f'all papers for training: {paper_number_for_training}')

if __name__ == '__main__':
    path = "/mnt/c/Users/hsluser/Desktop/2022_zhangchunpu/ChemECorpusLarge"
    main(path)



