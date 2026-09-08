import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.settings import settings
from src.llm.prompt_templates import build_prompt
from src.ingestion.document_loaders import load_pdf, load_document

class TestRAGOptimization(unittest.TestCase):
    def test_settings(self):
        self.assertEqual(settings.EMBEDDING_MODEL, "intfloat/multilingual-e5-small")
        self.assertEqual(settings.CHUNK_SIZE, 300)

    def test_prompt_building(self):
        prompt = build_prompt("Test context", "Test query")
        self.assertIn("Test context", prompt)
        self.assertIn("Test query", prompt)
        self.assertIn("TUYỆT ĐỐI KHÔNG chỉ cung cấp tên tệp", prompt)

    def test_document_loader_unsupported(self):
        with self.assertRaises(ValueError):
            load_document("nonexistent.unknown")

if __name__ == "__main__":
    unittest.main()
