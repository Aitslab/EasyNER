import time
import logging
from .tokenizers import TokenizerBase
from .writers import OutputWriterBase

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.processor")


class SplitterProcessor:
    def __init__(self, tokenizer, output_writer, config, progress_callback=None):
        self.tokenizer = tokenizer
        self.output_writer = output_writer
        self.config = config
        self.progress_callback = progress_callback  # Function to call for progress updates

        # Detect if tokenizer supports batch processing
        self.supports_batch_processing = hasattr(self.tokenizer, 'split_texts_batch')
        logger.info(f"Tokenizer batch processing support: {self.supports_batch_processing}")

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

        logger.debug(f"Processing batch {batch_idx} with {total_articles} articles")
        logger.debug(f"Using text field: {text_field} (pubmed={is_pubmed})")

        # Use batch processing if supported and enabled
        if self.supports_batch_processing and is_pubmed and total_articles > 1:
            logger.info(f"Using batch processing for {total_articles} PubMed documents")
            return self._process_batch_optimized(batch_idx, batch, text_field, full_articles)

        # Original single-document processing for backward compatibility
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
                logger.debug(f"Article {article_id} has empty {text_field} field")
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

                # Store the processed sentences
                processed_article["sentences"] = sentences
                processed_articles[article_id] = processed_article

                # Log sentence count at debug level
                logger.debug(
                    f"Article {article_id} processed: {len(sentences)} sentences extracted"
                )

            except Exception as e:
                logger.error(f"Error processing article {article_id}: {e}", exc_info=True)
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
        logger.debug(f"Writing {len(processed_articles)} articles to output")
        batch_idx, num_articles = self.output_writer.write(
            processed_articles, batch_idx, self.tokenizer.__class__.__name__
        )
        logger.debug(f"Completed batch {batch_idx}: {num_articles} articles written")

        # Final progress report to ensure 100% is reported
        if self.progress_callback:
            self.progress_callback(batch_idx, total_articles, total_articles)

        return batch_idx, num_articles

    def _process_batch_optimized(self, batch_idx, batch, text_field, full_articles):
        """Process multiple articles at once using batch tokenization"""
        start_time = time.time()
        processed_articles = {}
        total_articles = len(batch)

        # Collect all article texts and keep track of their IDs
        texts = []
        article_ids = []
        empty_articles = {}

        # First pass: collect texts and handle empty articles
        for article_id, article_data in batch.items():
            if full_articles:
                text = article_data.get(text_field, "")
                if not text:
                    # Handle empty articles immediately
                    empty_articles[article_id] = {
                        "title": article_data.get("title", ""),
                        "sentences": []
                    }
                    continue
            else:
                text = article_data
                if not text:
                    empty_articles[article_id] = {"sentences": []}
                    continue

            # Add non-empty texts to the batch
            texts.append(text)
            article_ids.append(article_id)

        # Process all texts in a single batch
        logger.debug(f"Processing batch of {len(texts)} texts (skipped {len(empty_articles)} empty)")

        if texts:  # Only process if we have non-empty texts
            # Process all texts in a single batch call
            try:
                all_sentences = self.tokenizer.split_texts_batch(texts)

                # Map results back to article IDs
                for i, (article_id, sentences) in enumerate(zip(article_ids, all_sentences)):
                    if full_articles:
                        article_data = batch[article_id]
                        # Format sentences as objects with "text" property for PubMed
                        formatted_sentences = [{"text": s} for s in sentences]
                        processed_articles[article_id] = {
                            "title": article_data.get("title", ""),
                            "sentences": formatted_sentences
                        }
                    else:
                        formatted_sentences = sentences
                        processed_articles[article_id] = {"sentences": formatted_sentences}

                    # Report progress periodically
                    if i % 100 == 0 or i == len(article_ids) - 1:
                        if self.progress_callback:
                            self.progress_callback(batch_idx, i + 1, len(article_ids))

            except Exception as e:
                logger.error(f"Error in batch processing: {e}", exc_info=True)
                # Fall back to individual processing if batch fails
                logger.warning("Falling back to individual document processing")
                return self.process_batch(batch_idx, batch, full_articles)

        # Add empty articles to the results
        processed_articles.update(empty_articles)

        # Write all processed articles
        batch_idx, num_articles = self.output_writer.write(
            processed_articles, batch_idx, self.tokenizer.__class__.__name__
        )

        processing_time = time.time() - start_time
        logger.info(f"Batch {batch_idx}: Processed {num_articles} articles in {processing_time:.2f}s using batch optimization")

        # Final progress report
        if self.progress_callback:
            self.progress_callback(batch_idx, total_articles, total_articles)

        return batch_idx, num_articles
