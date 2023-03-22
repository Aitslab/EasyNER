# coding=utf-8


def detokenize(token_label_pairs):
    labels = []
    words = []

    for token, label in token_label_pairs:
        if 'X' not in label:
            words.append(token)
            labels.append(label)
        else:
            word = words.pop(len(words)-1) + token[2:]
            words.append(word)

    return list(zip(labels, words))


def co_occurrence_extractor(label_word_pairs):
    entities = []
    entity = ""
    in_entity = False

    for label, word in label_word_pairs:

        if 'B' in label:
            entity = entity + word
            in_entity = True

        elif in_entity:
            if 'I' in label:
                entity = entity + " " + word
            elif 'O' in label:
                in_entity = False
                # TODO: format inside of entity e.g. " , ", " - ", etc.
                entity = entity.replace(' - ', '-')
                entity = entity.replace(' , ', ',')
                entities.append(entity)
                entity = ''

    return {
        "hasCoOccurrence": len(entities) >= 2,
        "entities": entities,
        "text": " ".join(list(map(lambda t: t[1], label_word_pairs))).
        replace(" .", ".").
        replace(" ,", ",").
        replace(" - ", "-").
        replace("( ", "(").
        replace(" )", ")").
        replace(" :", ":").
        replace(" ;", ";").
        replace(" !", "!").
        replace(" ?", "?")
    }


