from gensim.corpora.wikicorpus import extract_pages, filter_wiki
import argparse
import logging
from tqdm import tqdm

def extract_text_content(xml_dump):
    # article_count = 0
    with open('wiki.en.txt', 'w') as file:
        for title, content, pageid in tqdm(extract_pages(xml_dump)):
            try:
                file.write(filter_wiki(content).strip() + "\n")
                # article_count += 1
                # if article_count % 10000 == 0:
                #     logging.info(f'{article_count} articles processed')
            except Exception as e:
                logging.warning(str(e))

def main():
    parser = argparse.ArgumentParser(description='Extract the text contents of the pages from a Wikipedia dump XML file')
    parser.add_argument('dump_file',
                        help='Wikipedia dump XML file')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    extract_text_content('enwiki-latest-pages-articles.xml.bz2')

if __name__ == '__main__':
    main()