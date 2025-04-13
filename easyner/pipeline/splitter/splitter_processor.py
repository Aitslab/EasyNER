import time
import gc
import logging

from .tokenizers import TokenizerBase

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.processor")


class SplitterProcessor:
    def __init__(self, tokenizer: TokenizerBase, output_writer, config, progress_callback=None):
        self.tokenizer = tokenizer
        self.output_writer = output_writer
        self.config = config
        self.progress_callback = progress_callback
        self.worker_id = config.get("worker_id", "unknown")

        # Detect if tokenizer supports batch processing and batch generation
        self.supports_batch_processing = self.tokenizer.SUPPORTS_BATCH_PROCESSING
        self.supports_batch_generator = self.tokenizer.SUPPORTS_BATCH_GENERATOR
        logger.debug(
            f"[Worker {self.worker_id}] Tokenizer batch processing support: {self.tokenizer.SUPPORTS_BATCH_PROCESSING}"
        )
        logger.debug(
            f"[Worker {self.worker_id}] Tokenizer batch generator support: {self.tokenizer.SUPPORTS_BATCH_GENERATOR}"  # Log generator support
        )

        self.max_batch_size = self.config.get("max_tokenizer_batch_size", 5000)
        logger.debug(f"[Worker {self.worker_id}] Max tokenizer batch size: {self.max_batch_size}")

    def process_batch(self, batch_idx, batch, full_articles=None):
        """Process a batch of articles with more frequent progress reporting"""
        processed_articles = {}
        total_articles = len(batch)
        processed_count = 0
        start_batch_time = time.time()  # Track batch start time

        min_update_interval = 0.5
        last_update_time = time.time()
        report_interval = max(10, min(500, total_articles // 100))

        is_pubmed = self.config.get("pubmed_bulk", False)
        default_text_field = "abstract" if is_pubmed else "text"
        text_field = self.config.get("text_field", default_text_field)

        logger.debug(
            f"[Worker {self.worker_id}] Processing batch {batch_idx} with {total_articles} articles"
        )
        logger.debug(
            f"[Worker {self.worker_id}] Using text field: {text_field} (pubmed={is_pubmed})"
        )

        # --- Prioritize Batch Generator ---
        if self.tokenizer.SUPPORTS_BATCH_GENERATOR and is_pubmed and total_articles > 1:
            logger.debug(f"[Worker {self.worker_id}] Using batch generator for batch {batch_idx}")
            processed_articles = self._process_batch_generator(
                batch_idx, batch, text_field, full_articles
            )
        # --- Fallback to Standard Batch ---
        elif self.tokenizer.SUPPORTS_BATCH_PROCESSING and is_pubmed and total_articles > 1:
            logger.debug(
                f"[Worker {self.worker_id}] Using standard batch processing for batch {batch_idx}"
            )
            processed_articles = self._process_batch_optimized(
                batch_idx, batch, text_field, full_articles
            )
        # --- Fallback to Single Document Processing ---
        else:
            logger.debug(
                f"[Worker {self.worker_id}] Using single document processing for batch {batch_idx}"
            )
            # Use OrderedDict to maintain the original order of articles
            processed_articles = {}
            for article_id, article_data in batch.items():
                try:
                    text_to_split = article_data.get(text_field, "")
                    title = article_data.get("title", "")  # Ensure title is included

                    if text_to_split:
                        sentences = self.tokenizer.segement_sentences(text_to_split)
                        # Format output based on pubmed_bulk setting
                        if is_pubmed:
                            processed_articles[article_id] = {
                                "title": title,
                                "sentences": [{"text": s} for s in sentences],
                            }
                        else:
                            processed_articles[article_id] = {
                                "sentences": sentences,
                            }
                    else:
                        # Handle empty text
                        if is_pubmed:
                            processed_articles[article_id] = {
                                "title": title,
                                "sentences": [],
                            }
                        else:
                            processed_articles[article_id] = {
                                "sentences": [],
                            }

                    processed_count += 1
                    current_time = time.time()
                    if self.progress_callback and (
                        processed_count % report_interval == 0
                        or current_time - last_update_time > min_update_interval
                    ):
                        self.progress_callback(batch_idx, processed_count, total_articles)
                        last_update_time = current_time

                except Exception as e:
                    logger.error(
                        f"[Worker {self.worker_id}] Error processing article {article_id} in batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    # Ensure entry exists even on error
                    if is_pubmed:
                        processed_articles[article_id] = {
                            "title": article_data.get("title", ""),
                            "sentences": [],
                        }
                    else:
                        processed_articles[article_id] = {
                            "sentences": [],
                        }

        # --- Write Output ---
        logger.debug(
            f"[Worker {self.worker_id}] Writing {len(processed_articles)} articles from batch {batch_idx}"
        )
        write_start_time = time.time()
        written_batch_idx, num_articles_written = self.output_writer.write(
            processed_articles, batch_idx, self.tokenizer.__class__.__name__
        )
        write_duration = time.time() - write_start_time
        logger.debug(
            f"[Worker {self.worker_id}] Completed writing batch {batch_idx}: {num_articles_written} articles written in {write_duration:.2f}s"
        )

        # --- Final Progress & Stats ---
        batch_duration = time.time() - start_batch_time
        if self.progress_callback:
            self.progress_callback(batch_idx, total_articles, total_articles)  # Ensure 100%

        # Log batch completion statistics for the SummaryHandler
        stats = {
            "batch_complete": True,
            "batch_id": batch_idx,
            "articles": total_articles,
            "batch_size": total_articles,
            "processing_time": batch_duration,
            "worker_id": self.worker_id,
        }
        logger.info(f"Batch {batch_idx} complete.", extra={"statistics": stats})

        # Explicit garbage collection after processing and writing a batch
        del processed_articles
        del batch  # Assuming batch is not needed after this point
        gc.collect()
        logger.debug(
            f"[Worker {self.worker_id}] Garbage collection triggered after batch {batch_idx}"
        )

        return written_batch_idx, num_articles_written

    def _process_batch_generator(self, batch_idx, batch, text_field, full_articles):
        """Process multiple articles at once using the batch generator for memory efficiency"""
        start_time = time.time()
        processed_articles = {}
        total_articles = len(batch)
        texts_to_process = []
        article_ids_in_order = []
        empty_articles = {}  # Track articles with no text
        original_order = list(batch.keys())  # Preserve the original order of all articles

        # First pass: collect texts and IDs, handle empty/missing text
        for article_id, article_data in batch.items():
            text_to_split = article_data.get(text_field, "")
            title = article_data.get("title", "")  # Ensure title is included
            if text_to_split:
                texts_to_process.append(text_to_split)
                article_ids_in_order.append(article_id)
            else:
                # Store the title along with the empty sentences structure
                empty_articles[article_id] = {"title": title, "sentences": []}

        logger.debug(
            f"[Worker {self.worker_id}] Tokenizing {len(texts_to_process)} non-empty articles using batch generator."
        )

        # Process texts using the generator and zip with IDs
        try:
            # The generator yields lists of sentences, one list per input text
            sentence_lists_generator = self.tokenizer.segment_sentences_batch_generator(
                texts_to_process
            )

            for article_id, sentences in zip(article_ids_in_order, sentence_lists_generator):
                # Format output based on pubmed_bulk setting (assuming generator is used mainly for pubmed)
                # Get the original title from the input batch dictionary
                original_title = batch[article_id].get("title", "")
                processed_articles[article_id] = {
                    "title": original_title,
                    "sentences": [{"text": s} for s in sentences],
                }

            # Add articles with empty abstracts
            for article_id, article_data in empty_articles.items():
                processed_articles[article_id] = article_data

        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Error during batch generation for batch {batch_idx}: {e}",
                exc_info=True,
            )
            # Handle error: potentially mark all articles in this batch as failed or return partial results
            # For simplicity, returning currently processed + empty ones
            for article_id, article_data in empty_articles.items():
                if article_id not in processed_articles:
                    processed_articles[article_id] = article_data

        # Reconstruct the output in original order
        ordered_processed_articles = {
            article_id: processed_articles.get(article_id, {"sentences": []})
            for article_id in original_order
        }

        elapsed = time.time() - start_time
        logger.debug(
            f"[Worker {self.worker_id}] Batch {batch_idx} tokenization (generator) finished in {elapsed:.2f}s for {total_articles} articles."
        )
        return ordered_processed_articles

    def _process_batch_optimized(self, batch_idx, batch, text_field, full_articles):
        """Process multiple articles at once using standard batch tokenization"""
        start_time = time.time()
        processed_articles = {}
        total_articles = len(batch)
        texts_to_process = []
        article_ids_in_order = []
        empty_articles = {}  # Track articles with no text
        original_order = list(batch.keys())  # Preserve the original order of all articles

        # First pass: collect texts and IDs, handle empty/missing text
        for article_id, article_data in batch.items():
            text_to_split = article_data.get(text_field, "")
            title = article_data.get("title", "")  # Ensure title is included
            if text_to_split:
                texts_to_process.append(text_to_split)
                article_ids_in_order.append(article_id)
            else:
                # Store the title along with the empty sentences structure
                empty_articles[article_id] = {"title": title, "sentences": []}

        logger.debug(
            f"[Worker {self.worker_id}] Tokenizing {len(texts_to_process)} non-empty articles using standard batch."
        )

        # Process non-empty texts in a batch
        if texts_to_process:
            try:
                # This returns a list of lists of sentences
                results = self.tokenizer.segment_sentences_batch(texts_to_process)

                # Map results back to article IDs
                for article_id, sentences in zip(article_ids_in_order, results):
                    # Format output based on pubmed_bulk setting
                    title = batch[article_id].get("title", "")  # Ensure title is included
                    processed_articles[article_id] = {
                        "title": title,
                        "sentences": [{"text": s} for s in sentences],
                    }

            except Exception as e:
                logger.error(
                    f"[Worker {self.worker_id}] Error during standard batch processing for batch {batch_idx}: {e}",
                    exc_info=True,
                )
                # Handle error: Mark articles in this batch as failed or return partial results
                # Add successfully processed ones before error, if any, plus empty ones
                for article_id, article_data in empty_articles.items():
                    processed_articles[article_id] = article_data

        # Add articles with empty abstracts
        for article_id, article_data in empty_articles.items():
            processed_articles[article_id] = article_data

        # Reconstruct the output in original order
        ordered_processed_articles = {
            article_id: processed_articles.get(article_id, {"sentences": []})
            for article_id in original_order
        }

        elapsed = time.time() - start_time
        logger.debug(
            f"[Worker {self.worker_id}] Batch {batch_idx} tokenization (standard batch) finished in {elapsed:.2f}s for {total_articles} articles."
        )
        return ordered_processed_articles
