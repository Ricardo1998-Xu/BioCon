import ast
import json
from pathlib import Path

def is_valid_function(name):
    blacklist = []
    if name.startswith("test_"):
        return False

    if name in blacklist:
        return False

    return True


class FunctionExtractor(ast.NodeVisitor):

    def __init__(self, source, repo, file_path):

        self.source = source
        self.repo = repo
        self.file_path = file_path
        self.lines = source.splitlines()
        self.results = []
        self.current_class = None

    def get_code(self, node):
        return "\n".join(self.lines[node.lineno - 1: node.end_lineno])

    def get_signature(self, node):
        start = node.lineno - 1
        end = node.body[0].lineno - 1 if node.body else node.lineno

        return "\n".join(self.lines[start:end]).strip()

    def build_chunk(self, node):
        doc = ast.get_docstring(node)

        return {
            "repo": self.repo,
            "file": str(self.file_path),
            "module": str(self.file_path).replace("\\", "/"),
            "class": self.current_class,
            "function": node.name,
            "signature": self.get_signature(node),
            "docstring": doc,
            "code": self.get_code(node),
            "lineno": node.lineno,
            "end_lineno": node.end_lineno,
        }

    # ---------- class ----------
    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if is_valid_function(item.name):
                    self.results.append(self.build_chunk(item))

        self.generic_visit(node)
        self.current_class = prev_class

    # ---------- top-level functions ----------
    def visit_FunctionDef(self, node):
        if self.current_class is None:
            if is_valid_function(node.name):
                self.results.append(self.build_chunk(node))

        self.generic_visit(node)


def extract_functions(py_file, repo):
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as e:
        print("Parse failed:", py_file, e)
        return []

    extractor = FunctionExtractor(source, repo, py_file)
    extractor.visit(tree)

    return extractor.results


def code_chunks(repo_dir, out_dir):
    repo_root = Path(repo_dir)
    output_file = Path(out_dir)

    output_file.mkdir(parents=True, exist_ok=True)
    for repo_dir in repo_root.iterdir():
        all_chunks = []
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name
        for py_file in repo_dir.rglob("*.py"):
            chunks = extract_functions(py_file, repo_name)
            all_chunks.extend(chunks)

        print("{}-Total chunks:".format(repo_name), len(all_chunks))

        with open(output_file/ f"{repo_name}.json", "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    in_dir = "BioCon/raw/repos"
    out_dir = "BioCon/processed/code_chunks"
    code_chunks(in_dir, out_dir)
