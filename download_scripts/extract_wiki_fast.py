from gensim.corpora.wikicorpus import extract_pages, process_article
import argparse
import logging
from tqdm import tqdm

def extract_text_content(xml_dump):
    article_count = 0
    dewiki = MyWikiCorpus(xml_dump)
    with open('wiki.en.txt', 'w') as file:
        for tokens in tqdm(dewiki.get_texts()):
            try:
                file.write(" ".join(tokens).strip() + "\n")
                article_count += 1
                if article_count % 10000 == 0:
                    logging.info(f'{article_count} articles processed')
            except Exception as e:
                logging.warning(str(e))

class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=20, lemmatize=utils.has_pattern(), dictionary={}, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                positions += len(tokens)
                yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length

def main():
    parser = argparse.ArgumentParser(description='Extract the text contents of the pages from a Wikipedia dump XML file')
    parser.add_argument('dump_file',
                        help='Wikipedia dump XML file')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    extract_text_content('enwiki-latest-pages-articles.xml.bz2')

if __name__ == '__main__':
    main()