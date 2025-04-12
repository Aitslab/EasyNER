class SplitterProcessor:
    def __init__(self, tokenizer, output_writer, config):
        self.tokenizer = tokenizer
        self.output_writer = output_writer
        self.config = config

    def process_batch(self, batch_idx, batch, full_articles=None):
        articles = {}

        # Process each article in the batch
        for idx in batch:
            article = full_articles[idx] if full_articles else batch[idx]
            sentences = self.tokenizer.split_text(article["abstract"])

            articles[idx] = {
                "title": article["title"],
                "sentences": [{"text": sentence} for sentence in sentences],
            }

        # Write output
        self.output_writer.write(
            articles, batch_idx, tokenizer_name=self.tokenizer.__class__.__name__.lower()
        )

        return batch_idx
