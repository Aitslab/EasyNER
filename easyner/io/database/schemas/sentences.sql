CREATE TABLE IF NOT EXISTS sentences (
    article_id INTEGER,
    sentence_id INTEGER,
    text VARCHAR,
    PRIMARY KEY (article_id, sentence_id),
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);
