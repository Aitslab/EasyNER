import time
from .tokenizers import TokenizerBase
from .writers import OutputWriterBase


class SplitterProcessor:
    def __init__(self, tokenizer, output_writer, config, progress_callback=None):
        self.tokenizer = tokenizer
        self.output_writer = output_writer
        self.config = config
        self.progress_callback = progress_callback  # Function to call for progress updates

    def process_batch(self, batch_idx, batch, full_articles=None):
        """Process a batch of articles with more frequent progress reporting"""
        processed_articles = {}
        total_articles = len(batch)
        processed_count = 0

        # Set minimum update interval (in seconds) and article count interval
        min_update_interval = 0.5  # Send progress at least every 0.5 seconds
        last_update_time = time.time()
        report_interval = max(10, min(500, total_articles // 100))

        is_pubmed = self.config.get("pubmed_bulk", False)
        default_text_field = "abstract" if is_pubmed else "text"
        text_field = self.config.get("text_field", default_text_field)

        for article_id, article_data in batch.items():
            # Extract text field based on config
            if full_articles:
                # Extract text from the appropriate field (default to abstract for PubMed)
                text = article_data.get(text_field, "")

                # Create a structured output with metadata
                processed_article = {
                    # Preserve title and other metadata fields
                    "title": article_data.get("title", ""),
                    "sentences": [],
                }
            else:
                text = article_data
                processed_article = {"sentences": []}

            # Skip processing if text is empty
            if not text:
                # print(f"Warning: Article {article_id} has empty {text_field} field")
                processed_articles[article_id] = processed_article
                processed_count += 1
                continue

            # Process the article - split into sentences
            try:
                sentence_strings = self.tokenizer.split_text(text)

                # Format sentences as objects with "text" property for PubMed
                if is_pubmed:
                    sentences = [{"text": s} for s in sentence_strings]
                else:
                    sentences = sentence_strings

                processed_article["sentences"] = sentences
                processed_articles[article_id] = processed_article
            except Exception as e:
                print(f"Error processing article {article_id}: {e}")
                # Add empty list for failed articles to maintain structure
                processed_articles[article_id] = {
                    "title": article_data.get("title", "") if full_articles else "",
                    "sentences": [],
                }

            # Increment counter and report progress
            processed_count += 1

            # Report progress based on count or time interval
            current_time = time.time()
            if (
                processed_count % report_interval == 0
                or current_time - last_update_time >= min_update_interval
            ):
                if self.progress_callback:
                    self.progress_callback(batch_idx, processed_count, total_articles)
                    last_update_time = current_time

        # Write processed articles to output
        batch_idx, num_articles = self.output_writer.write(
            processed_articles, batch_idx, self.tokenizer.__class__.__name__
        )

        # Final progress report to ensure 100% is reported
        if self.progress_callback:
            self.progress_callback(batch_idx, total_articles, total_articles)

        return batch_idx, num_articles
