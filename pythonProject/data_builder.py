import os
import re

import spacy
from datasets import *
from bs4 import BeautifulSoup


class DataBuilder:

    def __init__(self, input_folder_path, output_folder_path, txt_files_already_exist=True):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        if not txt_files_already_exist:
            self.create_txt_files()

    def create_txt_files(self,):

        nlp = spacy.load("/mnt/c/Users/hsluser/appdata/local/programs/python/python38/lib/site-packages/en_core_sci_lg/en_core_sci_lg-0.4.0")
        nlp.max_length = 2000000

        for curDir, dirs, files in os.walk(self.input_folder_path):
            files = [file for file in files if (not file.startswith('.'))]
            file_number = len(files)
            if files:
                file_created = 0
                file_skipped = 0
                file_existed = 0
                print(f'\n\n now at {curDir}')
                journal = curDir.split('/')[-1]
                path = os.path.join(self.output_folder_path, journal)
                if not os.path.exists(path):
                    os.makedirs(path)
                for file in files:
                    file_name = file.split('.')[0]
                    if not os.path.exists(os.path.join(path, file_name + '.txt')):
                        file_path = os.path.join(curDir, file)
                        with open(file_path) as f:
                            xml_str = f.read()
                            soup = BeautifulSoup(xml_str, 'html')
                            contents = self._preprocess_text(soup, nlp)
                            if contents is None:
                                file_skipped += 1
                                print(f'\r file created: {file_created}, file existed: {file_existed}, file skipped: {file_skipped}, all: {file_number}', end='')
                                continue

                        with open(os.path.join(path, file_name+'.txt'), 'w') as f:
                            f.write(contents)
                            file_created += 1
                            print(f'\r file created: {file_created}, file existed: {file_existed}, file skipped: {file_skipped}, all: {file_number}', end='')
                    else:
                        file_existed += 1
                        print(f'\r file created: {file_created}, file existed: {file_existed}, file skipped: {file_skipped}, all: {file_number}', end='')
        return

    def load_data(self):

        file_ab_paths = []
        file_number = 0
        print('get file paths')
        for curDir, dirs, files in os.walk(self.output_folder_path):
            files = [file for file in files if not file.startswith('.')]
            if files:
                for file in files:
                    file_path = os.path.join(curDir, file)
                    file_ab_paths.append(file_path)
                    file_number += 1
                    print(f'\rfile prepared: {file_number}', end='')

        print('\nload data set ...')
        print('\n')

        dataset = load_dataset("text", data_files=file_ab_paths, split="train")
        return dataset, file_ab_paths

    @staticmethod
    def _preprocess_text(soup, nlp):

        contents = ""
        EXC_TAG = ["ce:footnote", "ce:label"]

        # decompose tags that we dont need
        for tag in EXC_TAG:
            e_tags = soup.find_all(tag)
            for e_tag in e_tags:
                e_tag.decompose()

        # deal with formulas
        ce_formula_list = soup.find_all("ce:formula")
        for ce_formula in ce_formula_list:
            ce_formula_text = ce_formula.text.replace("\n", "") + " "
            ce_formula.replace_with(ce_formula_text)

        tag_formula_list = soup.find_all("formula")
        # formula（for old formulas）
        for tag_formula in tag_formula_list:
            tag_formula_text = tag_formula.text.replace("\n", "") + " "
            tag_formula.replace_with(tag_formula_text)

        # extract abstract（skip to next file if there is no abstract）
        abst = soup.find("ce:abstract")
        if abst is None:
            return None
        abst_content = abst.find("ce:simple-para")

        if abst_content is None:
            return None

        if abst_content.string is None:
            contents += ''
        else:
            contents += abst_content.string

        # extract paragraphs
        sts = soup.find_all("ce:para")

        for st in sts:
            for tag in st.strings:
                contents += tag
            contents += " "

        # remove '\n'
        contents = contents.replace("\n", "")

        # sentence + '\n'
        doc = nlp(contents)
        contents_ = ""
        for sent in doc.sents:
            contents_ += sent.text
            contents_ += "\n"

        # fine tune
        contents_ = re.sub(' {2,}', ' ', contents_)

        return contents_


