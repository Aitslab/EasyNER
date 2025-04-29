# coding=utf-8

import spacy
from spacy.matcher import PhraseMatcher

def run_ner_with_spacy(model_name, vocab_path, entity_type, sentences):
    
    # Prepare spacy, if it is needed
    print("Running NER with spacy")
    nlp = spacy.load(model_name)
    
    terms = []
    with open(vocab_path) as f:
        for line in f:
            x = line.strip()
            terms.append(x)
    
    print("Phraselist complete")
    
    matcher = PhraseMatcher(nlp.vocab, attr = "LOWER")
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(entity_type, patterns)
    
    for i, sentence in enumerate(sentences):
        ner_class = entity_type
        
        doc = nlp(sentences["text"])
        if store_tokens == "yes":
            tokens = []
            # tokens_idxs = []  #uncomment if you want a list of token character offsets within the sentence
            for token in doc:
                tokens.append(token.text) #to get a list of tokens in the sentence
            # tokens_idxs.append(token.idx) #uncomment if you want a list of token character offsets within the sentence
            articles[pmid]["sentences"][i]["tokens"] = tokens

    
        entities = []
        spans = []
        matches = matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            ent = span.text
            entities.append(ent)
            first_char = span.start_char
            last_char = span.end_char - 1
            spans.append((first_char, last_char)) 


        articles[pmid]["sentences"][i]["NER class"] = ner_class
        articles[pmid]["sentences"][i]["entities"] = entities
        articles[pmid]["sentences"][i]["entity spans"] = spans

if __name__ == "__main__":
    pass




