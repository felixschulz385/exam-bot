import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.generation.template import TemplateProcessor


class FakeParagraph:
    def __init__(self, text):
        self.text = text


class FakeCell:
    def __init__(self, texts):
        self.paragraphs = [FakeParagraph(text) for text in texts]


class FakeRow:
    def __init__(self, texts):
        self.cells = [FakeCell(texts)]


class FakeTable:
    def __init__(self, texts):
        self.rows = [FakeRow(texts)]


class FakeDocument:
    def __init__(self):
        self.paragraphs = [FakeParagraph("Hello {{ name }}")]
        self.tables = [FakeTable(["{{ course }}"])]
        self.saved_to = None

    def save(self, output_path):
        self.saved_to = output_path


class TemplateProcessorTests(unittest.TestCase):
    def test_find_and_replace_placeholders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "template.docx"
            template_path.touch()
            processor = TemplateProcessor(template_path)
            doc = FakeDocument()

            placeholders = processor.find_placeholders(doc)
            processor.replace_placeholders(doc, {"name": "Felix", "course": "AHE"})

        self.assertEqual(sorted(placeholders), ["course", "name"])
        self.assertEqual(doc.paragraphs[0].text, "Hello Felix")
        self.assertEqual(doc.tables[0].rows[0].cells[0].paragraphs[0].text, "AHE")

    def test_process_template_uses_loaded_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "template.docx"
            template_path.touch()
            output_path = Path(temp_dir) / "processed.docx"
            processor = TemplateProcessor(template_path)
            fake_doc = FakeDocument()

            with mock.patch.object(processor, "load_template", return_value=fake_doc):
                result = processor.process_template({"name": "Felix", "course": "AHE"}, output_path=output_path)

        self.assertEqual(result, output_path)
        self.assertEqual(fake_doc.saved_to, output_path)


if __name__ == "__main__":
    unittest.main()
