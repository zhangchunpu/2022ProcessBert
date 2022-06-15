import os
import shutil

from bs4 import BeautifulSoup

def main(input_folder_path, output_folder_path):
    """
    This is a function to check what kinds of data are in database
    """
    for curDir, dirs, files in os.walk(input_folder_path):
        files = [file for file in files if not file.startswith('.')]
        if files:
            # 出力のフォルダを確保する
            folder = curDir.split('/')[-1]
            path = os.path.join(output_folder_path, folder)
            if not os.path.exists(path):
                os.makedirs(path)

            versions = {}
            n = 0
            file_number = len(files)
            print(f'now at {curDir}', end='\n')

            for file in files:
                if file.endswith('xml'):
                    #目標のファイルを取得して解析する
                    file_path = os.path.join(curDir, file)
                    with open(file_path) as f:
                        xml_str = f.read()
                        try:
                            soup = BeautifulSoup(xml_str, 'xml')
                            head = str(soup).strip()
                            version = head.split(' ')[1]
                            versions[file] = version
                        except:
                            shutil.copy(os.path.join(curDir, file), output_folder_path+f'plain_text_files/{folder}/{file}')
                        n += 1
                        print(f'\r{n}/{file_number}', end='')

            #write file with version information in a txt file
            with open(os.path.join(path, 'file_xml_version.txt'), 'w') as f:
                for file, version in versions.items():
                    f.write(file+': '+version+'\n')

    return

if __name__ == '__main__':
    main("/mnt/d/2022_zhangchunpu/ChemECorpusBase/", "/mnt/c/Users/hsluser/Desktop/data_check_result")

