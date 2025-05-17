"""Parse PubMed XML files to extract structured information.

Provides methods to extract various elements from PubMed XML files:
- PMIDs from different sections (articles, books, deleted citations)
- Article titles, abstracts, author information
- Custom elements via XPath queries
"""

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from lxml import etree


class PubMedXMLParser:
    """Parser for PubMed XML files.

    Provides methods to efficiently parse PubMed XML using DTD
    and direct paths to elements.

    Attributes:
        dtd_path (Path): Path to the DTD file for PubMed XML.
        dtd (etree.DTD): DTD object for validating XML content.

    """

    def __init__(self, dtd_path: Path = Path("./pubmed_230101.dtd")) -> None:
        """Initialize the parser with a DTD file.

        Args:
            dtd_path (Path): Path to the DTD file for PubMed XML.

        """
        self.dtd_path = (
            dtd_path if dtd_path.is_absolute() else Path(__file__).parent / dtd_path
        )
        self.dtd = etree.DTD(self.dtd_path)

    def load_document(self, file_path: str) -> etree.ElementBase:
        """Load a PubMed XML document from a file.

        Args:
            file_path (str): Path to the PubMed XML file.

        Returns:
            etree.ElementBase: The root element of the XML document.

        Raises:
            ValueError: If the file cannot be parsed or found.

        """
        try:
            parser = etree.XMLParser(dtd_validation=False, no_network=True)
            tree = etree.parse(file_path, parser)
            return tree.getroot()
        except etree.XMLSyntaxError as e:
            msg = f"Error parsing XML file: {e}"
            raise ValueError(msg) from e
        except FileNotFoundError:
            msg = f"File not found: {file_path}"
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Error processing file {file_path}: {e}"
            raise ValueError(msg) from e

    def validate_document(self, root: etree.ElementBase) -> bool:
        """Validate a PubMed XML document against the DTD.

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            bool: True if the document is valid, False otherwise.

        """
        try:
            return self.dtd.validate(root)
        except Exception:
            return False

    def extract_pmids_by_type(
        self,
        root: etree.ElementBase,
    ) -> tuple[list[int], list[int], list[int]]:
        """Extract all types of PMIDs from a PubMed XML document.

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            Tuple[List[int], List[int], List[int]]: Tuple containing article PMIDs,
                                                   book PMIDs, and deleted PMIDs.

        """
        article_pmids = self._extract_article_pmids(root)
        book_pmids = self._extract_book_pmids(root)
        deleted_pmids = self._extract_deleted_pmids(root)

        return article_pmids, book_pmids, deleted_pmids

    def _extract_deleted_pmids(self, root: etree.ElementBase) -> list[int]:
        """Extract deleted PMIDs from a DeleteCitation element.

        Uses direct XPath expressions to efficiently extract PMIDs from DeleteCitation elements.
        <PubmedArticleSet><DeleteCitation><PMID>

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            List[int]: List of deleted PMIDs.

        """
        deleted_pmids = []
        pmid_elements = root.findall(".//DeleteCitation/PMID")

        for pmid_element in pmid_elements:
            if pmid_element.text:
                deleted_pmids.append(int(pmid_element.text))

        return deleted_pmids

    def _extract_article_pmids(self, root: etree.ElementBase) -> list[int]:
        """Extract PMIDs from PubmedArticle elements.

        <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            List[int]: List of PMIDs extracted from the XML content.

        """
        pmids = []
        pmid_elements = root.findall(".//PubmedArticle/MedlineCitation/PMID")

        for pmid_element in pmid_elements:
            if pmid_element.text:
                pmids.append(int(pmid_element.text))

        return pmids

    def _parse_date_element(self, date_element: etree.ElementBase) -> Optional[date]:
        """Parse an XML date element directly to a date object.

        Extracts Year, Month, and Day child elements to construct a date object.

        Args:
            date_element (etree.ElementBase): XML element containing Year, Month, Day elements

        Returns:
            Optional[date]: A date object or None if parsing fails

        """
        if date_element is None:
            return None

        try:
            # Extract year, month, day text
            year_elem = date_element.find("Year")
            month_elem = date_element.find("Month")
            day_elem = date_element.find("Day")

            # Ensure we have at least a year
            if year_elem is None or not year_elem.text:
                return None

            # Convert to integers with defaults of 1 for missing month/day
            year = int(year_elem.text)
            month = (
                int(month_elem.text)
                if month_elem is not None and month_elem.text
                else 1
            )
            day = int(day_elem.text) if day_elem is not None and day_elem.text else 1

            # Create and return date object
            return date(year, month, day)

        except (ValueError, TypeError):
            # Handle invalid date values
            return None

    def _extract_book_pmids(self, root: etree.ElementBase) -> list[int]:
        """Extract PMIDs from PubmedBookArticle elements.

        <PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            List[int]: List of PMIDs extracted from the XML content.

        """
        pmids = []
        pmid_elements = root.findall(".//PubmedBookArticle/BookDocument/PMID")

        for pmid_element in pmid_elements:
            if pmid_element.text:
                pmids.append(int(pmid_element.text))

        return pmids

    def extract_elements_by_xpath(
        self,
        root: etree.ElementBase,
        xpath_query: str,
        convert_func=None,
    ) -> list[Any]:
        """Extract elements using an XPath query.

        Args:
            root (etree.ElementBase): Root element of the XML document.
            xpath_query (str): XPath query to find elements
            convert_func (callable, optional): Function to convert element text values

        Returns:
            List[Any]: List of extracted values

        """
        elements = root.xpath(xpath_query)

        results = []
        for element in elements:
            # Handle different element types
            value = element.text if hasattr(element, "text") else element

            if value is not None and convert_func:
                try:
                    value = convert_func(value)
                except (ValueError, TypeError):
                    # Skip conversion if it fails
                    pass

            if value is not None:
                results.append(value)

        return results

    def extract_elements_to_dict(
        self,
        root: etree.ElementBase,
        xpath_mappings: Mapping[str, str],
        converters: Optional[Mapping[str, callable]] = None,
    ) -> dict[str, list[Any]]:
        """Extract multiple elements using a dictionary of XPath queries.

        Args:
            root (etree.ElementBase): Root element of the XML document.
            xpath_mappings (dict): Dictionary mapping keys to XPath queries
            converters (dict, optional): Dictionary mapping keys to converter functions

        Returns:
            Dict[str, List[Any]]: Dictionary mapping keys to lists of extracted values

        """
        results = {}

        for key, xpath in xpath_mappings.items():
            elements = root.xpath(xpath)
            converter = converters.get(key) if converters else None

            values = []
            for element in elements:
                # Handle different element types
                value = element.text if hasattr(element, "text") else element

                if value is not None and converter:
                    try:
                        value = converter(value)
                    except (ValueError, TypeError):
                        # Skip conversion if it fails
                        pass

                if value is not None:
                    values.append(value)

            results[key] = values

        return results

    # Higher-level convenience methods
    def extract_pmids(self, root: etree.ElementBase) -> list[int]:
        """Extract all PMIDs (articles and books, excluding deleted) from a document.

        Args:
            root (etree.ElementBase): Root element of the XML document.

        Returns:
            List[int]: Combined list of article and book PMIDs

        """
        article_pmids, book_pmids, _ = self.extract_pmids_by_type(root)
        return article_pmids + book_pmids


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Use a sample file path - replace with actual path when testing
    sample_path = "/lunarc/nobackup/projects/snic2020-6-41/carl/data/pubmed_raw/rawpubmed25n0003.xml.gz"

    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = sample_path
        print(f"No file path provided, using example: {file_path}")
        print("You can provide a file path as a command-line argument")

    # Initialize the parser
    parser = PubMedXMLParser()

    try:
        # Load document once and reuse for all operations
        print(f"\nLoading document: {file_path}")
        root = parser.load_document(file_path)

        # Validate document
        is_valid = parser.validate_document(root)
        print(f"Document valid according to DTD: {is_valid}")

        # Extract PMIDs by type
        article_pmids, book_pmids, deleted_pmids = parser.extract_pmids_by_type(root)

        print(f"\nFound {len(article_pmids)} article PMIDs")
        print(f"Found {len(book_pmids)} book PMIDs")
        print(f"Found {len(deleted_pmids)} deleted PMIDs")

        # Print the first 5 PMIDs
        if article_pmids:
            print("First 5 article PMIDs:", article_pmids[:5])

        # Extract article titles using XPath
        titles = parser.extract_elements_by_xpath(
            root,
            ".//PubmedArticle/MedlineCitation/Article/ArticleTitle",
        )

        print(f"\nFound {len(titles)} article titles")
        if titles:
            print("First article title:", titles[0])

        # Extract multiple elements in one pass
        elements = parser.extract_elements_to_dict(
            root,
            {
                "titles": ".//PubmedArticle/MedlineCitation/Article/ArticleTitle",
                "abstracts": ".//PubmedArticle/MedlineCitation/Article/Abstract/AbstractText",
                "authors": ".//PubmedArticle/MedlineCitation/Article/AuthorList/Author/LastName",
            },
        )

        print("\nExtracted multiple elements:")
        print(f"- {len(elements['titles'])} titles")
        print(f"- {len(elements['abstracts'])} abstracts")
        print(f"- {len(elements['authors'])} author last names")

    except Exception as e:
        print(f"Error: {e}")
