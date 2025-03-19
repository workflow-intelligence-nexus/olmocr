import unittest

from olmocr.bench.tests import (
    BaselineTest,
    BasePDFTest,
    MathTest,
    TableTest,
    TestChecked,
    TestType,
    TextOrderTest,
    TextPresenceTest,
    ValidationError,
    normalize_text,
)


class TestNormalizeText(unittest.TestCase):
    """Test the normalize_text function"""

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized"""
        input_text = "This  has\tmultiple    spaces\nand\nnewlines"
        expected = "This has multiple spaces and newlines"
        self.assertEqual(normalize_text(input_text), expected)

    def test_character_replacement(self):
        """Test that fancy characters are replaced with ASCII equivalents"""
        input_text = "This has 'fancy' “quotes” and—dashes"
        expected = "This has 'fancy' \"quotes\" and-dashes"
        self.assertEqual(normalize_text(input_text), expected)

    def test_empty_input(self):
        """Test that empty input returns empty output"""
        self.assertEqual(normalize_text(""), "")


class TestBasePDFTest(unittest.TestCase):
    """Test the BasePDFTest class"""

    def test_valid_initialization(self):
        """Test that a valid initialization works"""
        test = BasePDFTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        self.assertEqual(test.pdf, "test.pdf")
        self.assertEqual(test.page, 1)
        self.assertEqual(test.id, "test_id")
        self.assertEqual(test.type, TestType.BASELINE.value)
        self.assertEqual(test.max_diffs, 0)
        self.assertIsNone(test.checked)
        self.assertIsNone(test.url)

    def test_empty_pdf(self):
        """Test that empty PDF raises ValidationError"""
        with self.assertRaises(ValidationError):
            BasePDFTest(pdf="", page=1, id="test_id", type=TestType.BASELINE.value)

    def test_empty_id(self):
        """Test that empty ID raises ValidationError"""
        with self.assertRaises(ValidationError):
            BasePDFTest(pdf="test.pdf", page=1, id="", type=TestType.BASELINE.value)

    def test_negative_max_diffs(self):
        """Test that negative max_diffs raises ValidationError"""
        with self.assertRaises(ValidationError):
            BasePDFTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value, max_diffs=-1)

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with self.assertRaises(ValidationError):
            BasePDFTest(pdf="test.pdf", page=1, id="test_id", type="invalid_type")

    def test_run_method_not_implemented(self):
        """Test that run method raises NotImplementedError"""
        test = BasePDFTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        with self.assertRaises(NotImplementedError):
            test.run("content")

    def test_checked_enum(self):
        """Test that checked accepts valid TestChecked enums"""
        test = BasePDFTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value, checked=TestChecked.VERIFIED)
        self.assertEqual(test.checked, TestChecked.VERIFIED)


class TestTextPresenceTest(unittest.TestCase):
    """Test the TextPresenceTest class"""

    def test_valid_present_test(self):
        """Test that a valid PRESENT test initializes correctly"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="test text")
        self.assertEqual(test.text, "test text")
        self.assertTrue(test.case_sensitive)
        self.assertIsNone(test.first_n)
        self.assertIsNone(test.last_n)

    def test_valid_absent_test(self):
        """Test that a valid ABSENT test initializes correctly"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ABSENT.value, text="test text", case_sensitive=False)
        self.assertEqual(test.text, "test text")
        self.assertFalse(test.case_sensitive)

    def test_empty_text(self):
        """Test that empty text raises ValidationError"""
        with self.assertRaises(ValidationError):
            TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="")

    def test_present_text_exact_match(self):
        """Test that PRESENT test returns True for exact match"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="target text")
        result, _ = test.run("This is some target text in a document")
        self.assertTrue(result)

    def test_present_text_not_found(self):
        """Test that PRESENT test returns False when text not found"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="missing text")
        result, explanation = test.run("This document doesn't have the target")
        self.assertFalse(result)
        self.assertIn("missing text", explanation)

    def test_present_text_with_max_diffs(self):
        """Test that PRESENT test with max_diffs handles fuzzy matching"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="target text", max_diffs=2)
        result, _ = test.run("This is some targett textt in a document")
        self.assertTrue(result)

    def test_absent_text_found(self):
        """Test that ABSENT test returns False when text is found"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ABSENT.value, text="target text")
        result, explanation = test.run("This is some target text in a document")
        self.assertFalse(result)
        self.assertIn("target text", explanation)

    def test_absent_text_found_diffs(self):
        """Test that ABSENT test returns False when text is found"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ABSENT.value, text="target text", max_diffs=2)
        result, explanation = test.run("This is some target text in a document")
        self.assertFalse(result)
        result, explanation = test.run("This is some targett text in a document")
        self.assertFalse(result)
        result, explanation = test.run("This is some targettt text in a document")
        self.assertFalse(result)
        result, explanation = test.run("This is some targetttt text in a document")
        self.assertTrue(result)

    def test_absent_text_not_found(self):
        """Test that ABSENT test returns True when text is not found"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ABSENT.value, text="missing text")
        result, _ = test.run("This document doesn't have the target")
        self.assertTrue(result)

    def test_case_insensitive_present(self):
        """Test that case_sensitive=False works for PRESENT test"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="TARGET TEXT", case_sensitive=False)
        result, _ = test.run("This is some target text in a document")
        self.assertTrue(result)

    def test_case_insensitive_absent(self):
        """Test that case_sensitive=False works for ABSENT test"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ABSENT.value, text="TARGET TEXT", case_sensitive=False)
        result, explanation = test.run("This is some target text in a document")
        self.assertFalse(result)

    def test_first_n_limit(self):
        """Test that first_n parameter works correctly"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="beginning", first_n=20)
        result, _ = test.run("beginning of text, but not the end")
        self.assertTrue(result)

        # Test that text beyond first_n isn't matched
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="end", first_n=20)
        result, _ = test.run("beginning of text, but not the end")
        self.assertFalse(result)

    def test_last_n_limit(self):
        """Test that last_n parameter works correctly"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="end", last_n=20)
        result, _ = test.run("beginning of text, but not the end")
        self.assertTrue(result)

        # Test that text beyond last_n isn't matched
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="beginning", last_n=20)
        result, _ = test.run("beginning of text, but not the end")
        self.assertFalse(result)

    def test_both_first_and_last_n(self):
        """Test that combining first_n and last_n works correctly"""
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="beginning", first_n=15, last_n=10)
        result, _ = test.run("beginning of text, middle part, but not the end")
        self.assertTrue(result)

        # Text only in middle shouldn't be found
        test = TextPresenceTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, text="middle", first_n=15, last_n=10)
        result, _ = test.run("beginning of text, middle part, but not the end")
        self.assertFalse(result)


class TestTextOrderTest(unittest.TestCase):
    """Test the TextOrderTest class"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="first text", after="second text")
        self.assertEqual(test.before, "first text")
        self.assertEqual(test.after, "second text")

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with self.assertRaises(ValidationError):
            TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, before="first text", after="second text")

    def test_empty_before(self):
        """Test that empty before text raises ValidationError"""
        with self.assertRaises(ValidationError):
            TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="", after="second text")

    def test_empty_after(self):
        """Test that empty after text raises ValidationError"""
        with self.assertRaises(ValidationError):
            TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="first text", after="")

    def test_correct_order(self):
        """Test that correct order returns True"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="first", after="second")
        result, _ = test.run("This has first and then second in correct order")
        self.assertTrue(result)

    def test_incorrect_order(self):
        """Test that incorrect order returns False"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="second", after="first")
        result, explanation = test.run("This has first and then second in correct order")
        self.assertFalse(result)

    def test_before_not_found(self):
        """Test that 'before' text not found returns False"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="missing", after="present")
        result, explanation = test.run("This text has present but not the other word")
        self.assertFalse(result)

    def test_after_not_found(self):
        """Test that 'after' text not found returns False"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="present", after="missing")
        result, explanation = test.run("This text has present but not the other word")
        self.assertFalse(result)

    def test_max_diffs(self):
        """Test that max_diffs parameter works correctly"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="first", after="second", max_diffs=1)
        result, _ = test.run("This has firsst and then secand in correct order")
        self.assertTrue(result)

    def test_multiple_occurrences(self):
        """Test that multiple occurrences are handled correctly"""
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="target", after="target")
        result, _ = test.run("This has target and then target again")
        self.assertTrue(result)

        # Test reverse direction fails
        test = TextOrderTest(pdf="test.pdf", page=1, id="test_id", type=TestType.ORDER.value, before="B", after="A")
        result, _ = test.run("A B A B")  # A comes before B, but B also comes before second A
        self.assertTrue(result)


class TestTableTest(unittest.TestCase):
    """Test the TableTest class"""

    def setUp(self):
        """Set up test fixtures"""
        self.markdown_table = """
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3  |
"""

        self.html_table = """
<table>
  <tr>
    <th>Header 1</th>
    <th>Header 2</th>
    <th>Header 3</th>
  </tr>
  <tr>
    <td>Cell A1</td>
    <td>Cell A2</td>
    <td>Cell A3</td>
  </tr>
  <tr>
    <td>Cell B1</td>
    <td>Cell B2</td>
    <td>Cell B3</td>
  </tr>
</table>
"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="target cell")
        self.assertEqual(test.cell, "target cell")
        self.assertEqual(test.up, "")
        self.assertEqual(test.down, "")
        self.assertEqual(test.left, "")
        self.assertEqual(test.right, "")
        self.assertEqual(test.top_heading, "")
        self.assertEqual(test.left_heading, "")

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with self.assertRaises(ValidationError):
            TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, cell="target cell")

    def test_parse_markdown_tables(self):
        """Test markdown table parsing"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        tables = test.parse_markdown_tables(self.markdown_table)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].shape, (3, 3))  # 3 rows, 3 columns
        self.assertEqual(tables[0][0, 0], "Header 1")
        self.assertEqual(tables[0][1, 1], "Cell A2")
        self.assertEqual(tables[0][2, 2], "Cell B3")

    def test_parse_html_tables(self):
        """Test HTML table parsing"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        tables = test.parse_html_tables(self.html_table)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].shape, (3, 3))  # 3 rows, 3 columns
        self.assertEqual(tables[0][0, 0], "Header 1")
        self.assertEqual(tables[0][1, 1], "Cell A2")
        self.assertEqual(tables[0][2, 2], "Cell B3")

    def test_match_cell(self):
        """Test finding a cell in a table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

    def test_cell_not_found(self):
        """Test cell not found in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Missing Cell")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("No cell matching", explanation)

    def test_up_relationship(self):
        """Test up relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", up="Header 2")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect up relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", up="Wrong Header")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_down_relationship(self):
        """Test down relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", down="Cell B2")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect down relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", down="Wrong Cell")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_left_relationship(self):
        """Test left relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", left="Cell A1")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect left relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", left="Wrong Cell")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_right_relationship(self):
        """Test right relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", right="Cell A3")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect right relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", right="Wrong Cell")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_top_heading_relationship(self):
        """Test top_heading relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell B2", top_heading="Header 2")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect top_heading relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell B2", top_heading="Wrong Header")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_left_heading_relationship(self):
        """Test left_heading relationship in table"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A3", left_heading="Cell A1")
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test incorrect left_heading relationship
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A3", left_heading="Wrong Cell")
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_multiple_relationships(self):
        """Test multiple relationships in table"""
        test = TableTest(
            pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", up="Header 2", down="Cell B2", left="Cell A1", right="Cell A3"
        )
        result, _ = test.run(self.markdown_table)
        self.assertTrue(result)

        # Test one incorrect relationship
        test = TableTest(
            pdf="test.pdf",
            page=1,
            id="test_id",
            type=TestType.TABLE.value,
            cell="Cell A2",
            up="Header 2",
            down="Cell B2",
            left="Wrong Cell",  # This is incorrect
            right="Cell A3",
        )
        result, explanation = test.run(self.markdown_table)
        self.assertFalse(result)
        self.assertIn("doesn't match expected", explanation)

    def test_no_tables_found(self):
        """Test behavior when no tables are found"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        result, explanation = test.run("This is plain text with no tables")
        self.assertFalse(result)
        self.assertEqual(explanation, "No tables found in the content")

    def test_fuzzy_matching(self):
        """Test fuzzy matching with max_diffs"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2", max_diffs=1)
        # Create table with slightly misspelled cell
        misspelled_table = self.markdown_table.replace("Cell A2", "Cel A2")
        result, _ = test.run(misspelled_table)
        self.assertTrue(result)

    def test_with_stripped_content(self):
        """Test table parsing with stripped content"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Strip all leading/trailing whitespace from the markdown table
        stripped_table = self.markdown_table.strip()
        result, explanation = test.run(stripped_table)
        self.assertTrue(result, f"Table test failed with stripped content: {explanation}")

    def test_table_at_end_of_file(self):
        """Test that a table at the very end of the file is correctly detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Create content with text followed by a table at the very end with no trailing newline
        content_with_table_at_end = "Some text before the table.\n" + self.markdown_table.strip()
        result, explanation = test.run(content_with_table_at_end)
        self.assertTrue(result, f"Table at end of file not detected: {explanation}")

    def test_table_at_end_with_no_trailing_newline(self):
        """Test that a table at the end with no trailing newline is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Remove the trailing newline from the markdown table
        content_without_newline = self.markdown_table.rstrip()
        result, explanation = test.run(content_without_newline)
        self.assertTrue(result, f"Table without trailing newline not detected: {explanation}")

    def test_table_at_end_with_extra_spaces(self):
        """Test that a table at the end with extra spaces is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Add extra spaces to the end of lines in the table
        lines = self.markdown_table.split("\n")
        content_with_extra_spaces = "\n".join([line + "   " for line in lines])
        result, explanation = test.run(content_with_extra_spaces)
        self.assertTrue(result, f"Table with extra spaces not detected: {explanation}")

    def test_table_at_end_with_mixed_whitespace(self):
        """Test that a table at the end with mixed whitespace is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Add various whitespace characters to the table
        content_with_mixed_whitespace = "Some text before the table.\n" + self.markdown_table.strip() + "  \t  "
        result, explanation = test.run(content_with_mixed_whitespace)
        self.assertTrue(result, f"Table with mixed whitespace not detected: {explanation}")

    def test_malformed_table_at_end(self):
        """Test that a slightly malformed table at the end is still detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Create a table with irregular pipe placement at the end
        malformed_table = """
Some text before the table.
| Header 1 | Header 2 | Header 3
| -------- | -------- | --------
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3"""
        result, explanation = test.run(malformed_table)
        self.assertTrue(result, f"Malformed table at end not detected: {explanation}")

    def test_incomplete_table_at_end(self):
        """Test that an incomplete table at the end still gets detected if it contains valid rows"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Missing the separator row
        incomplete_table = """
Some text before the table.
| Header 1 | Header 2 | Header 3 |
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3  |"""
        result, explanation = test.run(incomplete_table)
        self.assertTrue(result, f"Incomplete table at end not detected: {explanation}")

    def test_table_with_excessive_blank_lines_at_end(self):
        """Test that a table followed by many blank lines is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Add many blank lines after the table
        table_with_blanks = self.markdown_table + "\n\n\n\n\n\n\n\n\n\n"
        result, explanation = test.run(table_with_blanks)
        self.assertTrue(result, f"Table with blank lines at end not detected: {explanation}")

    def test_table_at_end_after_long_text(self):
        """Test that a table at the end after a very long text is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Create a very long text before the table
        long_text = "Lorem ipsum dolor sit amet, " * 100
        content_with_long_text = long_text + "\n" + self.markdown_table.strip()
        result, explanation = test.run(content_with_long_text)
        self.assertTrue(result, f"Table after long text not detected: {explanation}")

    def test_valid_table_at_eof_without_newline(self):
        """Test that a valid table at EOF without a trailing newline is detected"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="Cell A2")
        # Valid table but without trailing newline at the very end of the file
        valid_table_eof = """
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3  |""".strip()
        result, explanation = test.run(valid_table_eof)
        self.assertTrue(result, f"Valid table at EOF without newline not detected: {explanation}")

    def test_normalizing(self):
        table = """| Question - – Satisfaction on scale of 10 | Response | Resident Sample | Business Sample |
|----------------------------------------|----------|----------------|-----------------|
| Planning for and managing residential, commercial and industrial development | Rating of 8, 9 or 10 | 13% | 11% |
| | Average rating | 6.4 | 5.7 |
| | Don’t know responses | 11% | 6% |
| Environmental protection, support for green projects (e.g. green grants, building retrofits programs, zero waste) | Rating of 8, 9 or 10 | 35% | 34% |
| | Average rating | 8.0 | 7.5 |
| | Don’t know responses | 8% | 6% |
| Providing and maintaining parks and green spaces | Rating of 8, 9 or 10 | 42% | 41% |
| | Average rating | 7.7 | 7.3 |
| | Don’t know responses | 1% | 1% |"""
        test = TableTest(pdf="test.pdf", page=1, id="test_id", type=TestType.TABLE.value, cell="6%", top_heading="Business\nSample")
        result, explanation = test.run(table)
        self.assertTrue(result, explanation)


class TestBaselineTest(unittest.TestCase):
    """Test the BaselineTest class"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value, max_repeats=50)
        self.assertEqual(test.max_repeats, 50)

    def test_non_empty_content(self):
        """Test that non-empty content passes"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        result, _ = test.run("This is some normal content")
        self.assertTrue(result)

    def test_empty_content(self):
        """Test that empty content fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        result, explanation = test.run("   \n\t  ")
        self.assertFalse(result)
        self.assertIn("no alpha numeric characters", explanation)

    def test_repeating_content(self):
        """Test that highly repeating content fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value, max_repeats=2)
        # Create highly repeating content - repeat "abc" many times
        repeating_content = "abc" * 10
        result, explanation = test.run(repeating_content)
        self.assertFalse(result)
        self.assertIn("repeating", explanation)

    def test_content_with_disallowed_characters(self):
        """Test that content with disallowed characters fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        result, explanation = test.run("This has Chinese characters: 你好")
        self.assertFalse(result)
        self.assertIn("disallowed characters", explanation)

    def test_content_with_emoji(self):
        """Test that content with emoji fails"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        result, explanation = test.run("This has emoji: 😊")
        self.assertFalse(result)
        self.assertIn("disallowed characters", explanation)
        self.assertIn("😊", explanation)

    def test_content_with_mandarin(self):
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        result, explanation = test.run("asdfasdfas維基百科/中文asdfw")
        self.assertFalse(result)
        self.assertIn("disallowed characters", explanation)

    def test_valid_content(self):
        """Test that valid content passes all checks"""
        test = BaselineTest(pdf="test.pdf", page=1, id="test_id", type=TestType.BASELINE.value)
        content = "This is some normal content with proper English letters and no suspicious repetition."
        result, _ = test.run(content)
        self.assertTrue(result)


class TestMathTest(unittest.TestCase):
    """Test the MathTest class"""

    def test_valid_initialization(self):
        """Test that valid initialization works"""
        try:
            test = MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="a + b = c")
            self.assertEqual(test.math, "a + b = c")
        except Exception as e:
            self.fail(f"Valid initialization failed with: {e}")

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValidationError"""
        with self.assertRaises(ValidationError):
            MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.PRESENT.value, math="a + b = c")

    def test_empty_math(self):
        """Test that empty math raises ValidationError"""
        with self.assertRaises(ValidationError):
            MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="")

    def test_exact_math_match(self):
        """Test exact match of math equation"""
        try:
            test = MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="a + b = c")

            # Test content with exact math match
            content = "Here is an equation: $$a + b = c$$"
            result, _ = test.run(content)
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Test failed with: {e}")

    def test_rendered_math_match(self):
        """Test rendered match of math equation"""
        try:
            test = MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="a + b = c")

            # Test content with different but equivalent math
            content = "Here is an equation: $$a+b=c$$"
            result, _ = test.run(content)
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Test failed with: {e}")

    def test_no_math_match(self):
        """Test no match of math equation"""
        try:
            test = MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="a + b = c")

            # Test content with no matching math
            content = "Here is an equation: $$x + y = z$$"
            result, explanation = test.run(content)
            self.assertFalse(result)
            self.assertIn("No match found", explanation)
        except Exception as e:
            self.fail(f"Test failed with: {e}")

    def test_different_math_delimiters(self):
        """Test different math delimiters"""
        try:
            test = MathTest(pdf="test.pdf", page=1, id="test_id", type=TestType.MATH.value, math="a + b = c")

            # Test different delimiters
            delimiters = [
                "$$a + b = c$$",  # $$...$$
                "$a + b = c$",  # $...$
                "\\(a + b = c\\)",  # \(...\)
                "\\[a + b = c\\]",  # \[...\]
            ]

            for delim in delimiters:
                content = f"Here is an equation: {delim}"
                result, _ = test.run(content)
                self.assertTrue(result)
        except Exception as e:
            self.fail(f"Test failed with: {e}")


if __name__ == "__main__":
    unittest.main()
