import sys
from datetime import date
from pathlib import Path

import pytest
from lxml import etree

from easyner.pipeline.pubmed.xml import PubMedXMLParser


class TestPubMedXMLParser:
    @pytest.fixture
    def parser(self):
        # Initialize the parser with a mock DTD path
        # The DTD validation isn't needed for these unit tests
        return PubMedXMLParser()

    def test_parse_date_element_complete_date(self, parser) -> None:
        # Test with a complete date (year, month, day)
        date_xml = """
            <ArticleDate DateType="Electronic">
                <Year>2015</Year>
                <Month>08</Month>
                <Day>01</Day>
            </ArticleDate>
        """
        date_element = etree.fromstring(date_xml)
        result = parser._parse_date_element(date_element)

        assert result is not None
        assert isinstance(result, date)
        assert result.year == 2015
        assert result.month == 8
        assert result.day == 1

    def test_parse_date_element_year_only(self, parser) -> None:
        # Test with only a year
        date_xml = """
            <ArticleDate DateType="Electronic">
                <Year>2015</Year>
            </ArticleDate>
        """
        date_element = etree.fromstring(date_xml)
        result = parser._parse_date_element(date_element)

        assert result is not None
        assert isinstance(result, date)
        assert result.year == 2015
        assert result.month == 1  # Default month
        assert result.day == 1  # Default day

    def test_parse_date_element_year_month(self, parser) -> None:
        # Test with year and month but no day
        date_xml = """
            <ArticleDate DateType="Electronic">
                <Year>2015</Year>
                <Month>08</Month>
            </ArticleDate>
        """
        date_element = etree.fromstring(date_xml)
        result = parser._parse_date_element(date_element)

        assert result is not None
        assert isinstance(result, date)
        assert result.year == 2015
        assert result.month == 8
        assert result.day == 1  # Default day

    def test_parse_date_element_none(self, parser) -> None:
        # Test with None input
        result = parser._parse_date_element(None)
        assert result is None

    def test_parse_date_element_invalid(self, parser) -> None:
        # Test with invalid date values
        date_xml = """
            <ArticleDate DateType="Electronic">
                <Year>invalid</Year>
                <Month>08</Month>
                <Day>01</Day>
            </ArticleDate>
        """
        date_element = etree.fromstring(date_xml)
        result = parser._parse_date_element(date_element)
        assert result is None
