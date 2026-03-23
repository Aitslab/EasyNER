import argparse
from collections import defaultdict
import time
from typing import Dict, List, Tuple

from bioc import pubtator
from flair.data import Sentence
from flair.models.prefixed_tagger import PrefixedSequenceTagger
from flair.models.entity_mention_linking import EntityMentionLinker
from flair.splitter import SciSpacySentenceSplitter, SentenceSplitter

ENTITY_TYPES = ("disease", "chemical", "gene", "species")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Hunflair2 on documents in PubTator format")
    parser.add_argument(
        "--input",
        type=str,
        default="./annotations/raw/tmvar_v3_text.txt",
        help="Raw (w/o annotation) file in PubTator format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./annotations/hunflair2/tmvar_v3.txt",
        help="File with Hunflair2 annotations",
    )
    parser.add_argument("--entity_types", nargs="*", default=["gene"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sentences to process in one go",
    )
    return parser.parse_args()


def load_documents(path: str) -> Dict[str, pubtator.PubTator]:
    documents = {}
    with open(path) as fp:
        for d in pubtator.load(fp):
            documents[d.pmid] = d
    return documents


def get_document_text(document: pubtator.PubTator) -> str:
    text = ""
    if document.title is not None:
        text += document.title
    if document.abstract is not None:
        text += " "
        text += document.abstract
    return text


def get_sentences(
    documents: Dict[str, pubtator.PubTator], splitter: SentenceSplitter
) -> List[Tuple[str, Sentence]]:
    sentences = []
    for pmid, document in documents.items():
        for s in splitter.split(get_document_text(document)):
            sentences.append((pmid, s))
    return sentences


def run_ner(
    tagger: PrefixedSequenceTagger,
    documents: Dict[str, pubtator.PubTator],
    batch_size: int,
    verbose: bool = True
):
    splitter = SciSpacySentenceSplitter()

    print("- start entity recognition")
    start = time.time()

    pmid_sentence = get_sentences(documents=documents, splitter=splitter)
    pmids, sentences = zip(*pmid_sentence)
    sentences = list(sentences)

    tagger.predict(sentences, mini_batch_size=batch_size, verbose=verbose)
    elapsed = round(time.time() - start, 2)
    print(f"- entity recognition took: {elapsed}s")

    for pmid, sentence in zip(pmids, sentences):
        for span in sentence.get_spans("ner"):
            annotation = pubtator.PubTatorAnn(
                pmid=pmid,
                text=span.text,
                start=span.start_position + sentence.start_position,
                end=span.end_position + sentence.start_position,
                type=span.tag.lower(),
                id=-1,
            )
            documents[pmid].add_annotation(annotation)

    for _, document in documents.items():
        document.annotations = sorted(document.annotations, key=lambda x: x.start)


def run_nen(
    linkers: Dict[str, EntityMentionLinker],
    documents: Dict[str, pubtator.PubTator],
    batch_size: int,
):
    print("- start entity linking")
    start = time.time()

    texts = {
        pmid: Sentence(get_document_text(document))
        for pmid, document in documents.items()
    }

    linker = list(linkers.values())[0]
    linker.preprocessor.initialize(list(texts.values()))
    for other_linker in list(linkers.values())[1:]:
        other_linker.preprocessor.abbreviation_dict = (
            linker.preprocessor.abbreviation_dict
        )

    for entity_type, linker in linkers.items():
        annotations = []
        annotations_text = []
        for pmid, document in documents.items():
            for a in document.annotations:
                if a.type.lower() == entity_type:
                    annotations.append(a)
                    annotations_text.append(
                        linker.preprocessor.process_mention(
                            entity_mention=a.text, sentence=texts[pmid]
                        )
                    )

        for i in range(0, len(annotations), batch_size):
            batch_annotations = annotations[i : i + batch_size]
            batch_annotations_text = annotations_text[i : i + batch_size]

            batch_candidates = linker.candidate_generator.search(
                entity_mentions=batch_annotations_text, top_k=1
            )

            assert len(batch_annotations_text) == len(batch_candidates), (
                f"# of mentions ({len(batch_annotations_text)}) !="
                + f" # of search results ({len(batch_candidates)})!"
            )

            for a, mention_candidates in zip(batch_annotations, batch_candidates):
                if len(mention_candidates) > 0:
                    top_candidate = mention_candidates[0]
                    a.id = top_candidate[0]

    elapsed = round(time.time() - start, 2)
    print(f"- entity linking took: {elapsed}s")


def main(args: argparse.Namespace):
    assert (
        len(args.entity_types) > 0
    ), "You must provide at least one entity type `--entity_types`"
    assert all(
        et in ENTITY_TYPES for et in args.entity_types
    ), f"There are invalid entity types. All must be one one of: {ENTITY_TYPES}"

    print("Start predicting with Hunflair2:")
    print(f"- input file: {args.input}")
    print(f"- output file: {args.output}")

    documents = load_documents(args.input)

    print("- load entity recognition model")
    tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")
    run_ner(tagger=tagger, documents=documents, batch_size=args.batch_size)

    # print(f"- load entity linking models: {args.entity_types}")
    # linkers = {et: EntityMentionLinker.load(f"{et}-linker") for et in args.entity_types}
    # run_nen(linkers=linkers, documents=documents, batch_size=args.batch_size)

    with open(args.output, "w") as fp:
        pubtator.dump(documents.values(), fp)

    print("- done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
