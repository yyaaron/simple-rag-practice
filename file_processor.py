"""
Download file and process it for later use
"""
import os
import random
import re

import requests
import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English


class FileDownloader:
    file_name = 'human-nutrition-text.pdf'
    download_url = 'https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf'

    def __init__(self, file_name: str = 'human-nutrition-text.pdf',
                 download_url: str = 'https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf'):
        self.download_url = download_url
        self.file_name = file_name

    @classmethod
    def download_if_not_exist(cls, file_name: str, download_url: str):
        if file_name == "" or file_name is None:
            file_name = cls.file_name

        if download_url == "" or download_url is None:
            download_url = cls.download_url

        if not os.path.exists(file_name):
            # download file from download_url
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(file_name, "wb") as file:
                    file.write(response.content)

                print("[INFO]File downloaded with name", file_name)
            else:
                print("[INFO]Exception occurs while downloading")
        else:
            print("[INFO]File", file_name, "exists, ignore downloading")


class FilePreprocessor:
    num_sentence_chunk_size = 10

    def __init__(self):
        pass

    @classmethod
    def clean_text(cls, text) -> str:
        return text.replace("\n", " ").strip()

    @classmethod
    def text_to_sentences(cls, text) -> list:
        nlp = English()
        nlp.add_pipe("sentencizer")

        doc = nlp(text)
        return list(doc.sents)

    @classmethod
    def split_list(cls, input_list: list, slice_size: int) -> list[list[str]]:
        return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]

    @classmethod
    def open_and_read_pdf(cls, file_path: str) -> list[dict]:
        doc = fitz.open(file_path)
        pages_and_texts = []
        print("[INFO]Loading content into memory...")
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = cls.clean_text(text)
            pages_and_texts.append(
                {"page_number": page_number - 41,
                 "page_char_count": len(text),
                 "page_word_count": len(text.split(" ")),
                 "page_sentence_count_raw": len(text.split(". ")),
                 "page_token_count": len(text) / 4,
                 "text": text
                 })

        print("[INFO]Make sentences into chunks...")
        for item in tqdm(pages_and_texts):
            item["sentences"] = cls.text_to_sentences(item["text"])
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
            item["sentence_chunks"] = cls.split_list(input_list=item["sentences"],
                                                     slice_size=cls.num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        print(f"[INFO]Finish loading file({file_path})")
        return pages_and_texts

    @classmethod
    def pages_and_chunks(cls, pages_and_texts: list[dict], min_token_len: int = 30) -> list[dict]:
        print(f"[INFO]Divide chunks into specified-size pieces and filter out small chunks(size <= {min_token_len})...")
        pages_and_chunks = []
        for item in tqdm(pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                # ".A" -> ". A" for any full-stop/capital letter combo
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

                chunk_token_count = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters
                if chunk_token_count <= min_token_len:
                    continue  # small token sentences are always headers and footers, so filter them out as are useless
                chunk_dict["chunk_token_count"] = chunk_token_count
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])

                pages_and_chunks.append(chunk_dict)

        print("[INFO]Finish processing chunks for each pages")

        return pages_and_chunks

