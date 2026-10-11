import ast
from pathlib import Path


def test_no_product_imports():
    for path in Path("inference_kit").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split('.')[0] in {'core', 'app', 'server', 'riftbound', 'reserve', 'cardscanner'} for n in names), path
