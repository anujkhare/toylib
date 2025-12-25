"""Tests for bundler."""

import ast

from . import bundler as bundler


class TestImportRemover:
    """Tests for ImportRemover class."""

    def test_simple_import_bundled(self):
        """Test simple import statement for bundled package."""
        source = "import mypackage"
        tree = ast.parse(source)

        remover = bundler.ImportRemover({"mypackage"})
        tree = remover.visit(tree)

        # Import should be removed
        assert len(tree.body) == 0
        # Should be classified as bundled
        assert len(remover.imports_to_bundle) == 1
        assert remover.imports_to_bundle[0].package == "mypackage"
        assert remover.imports_to_bundle[0].name == "mypackage"
        # Should track imported name
        assert "mypackage" in remover.imported_names

    def test_simple_import_external(self):
        """Test simple import statement for external package."""
        source = "import numpy"
        tree = ast.parse(source)

        remover = bundler.ImportRemover({"mypackage"})
        tree = remover.visit(tree)

        # Import should be removed
        assert len(tree.body) == 0
        # Should have import statement string
        assert len(remover.external_import_statements) == 1
        assert remover.external_import_statements[0] == "import numpy"
        # Should not be in bundled imports
        assert len(remover.imports_to_bundle) == 0

    def test_dotted_import(self):
        """Test dotted import statement."""
        source = "import mypackage.submodule"
        tree = ast.parse(source)

        remover = bundler.ImportRemover({"mypackage"})
        tree = remover.visit(tree)

        assert len(remover.imports_to_bundle) == 1
        assert remover.imports_to_bundle[0].package == "mypackage"
        assert remover.imports_to_bundle[0].name == "mypackage.submodule"
        assert "mypackage.submodule" in remover.imported_names

    def test_mixed_bundled_and_external(self):
        """Test code with both bundled and external imports."""
        source = """
import mypackage
import numpy
from mypackage.utils import helper
from pandas import DataFrame
def foo(): pass
        """
        tree = ast.parse(source)

        remover = bundler.ImportRemover({"mypackage"})
        tree = remover.visit(tree)

        # All imports should be removed from AST
        assert len(tree.body) == 1

        # Should have 2 bundled imports
        assert len(remover.imports_to_bundle) == 2
        bundled_modules = [imp.name for imp in remover.imports_to_bundle]
        assert "mypackage" in bundled_modules
        assert "mypackage.utils.helper" in bundled_modules

        # Should have 2 external import statements
        assert len(remover.external_import_statements) == 2


class TestAttributeStripper:
    """Tests for AttributeStripper class."""

    def test_simple_attribute_access(self):
        """Test: mypackage.foo -> foo."""
        source = "result = mypackage.foo()"
        tree = ast.parse(source)

        stripper = bundler.AttributeStripper({"mypackage"})
        tree = stripper.visit(tree)

        # Convert back to source and check
        result = ast.unparse(tree)
        assert "mypackage" not in result
        assert "result = foo()" in result

    def test_nested_attribute_access(self):
        """Test: mypackage.sub.func -> func."""
        source = "result = mypackage.sub.func()"
        tree = ast.parse(source)

        stripper = bundler.AttributeStripper({"mypackage"})
        tree = stripper.visit(tree)

        result = ast.unparse(tree)
        assert "mypackage" not in result
        assert "result = func()" in result


class TestBundlerV2:
    def test_process_file(self):
        """Basic smoke test for the bundler."""
        obj = bundler.Bundler(
            packages_to_bundle=["internal1", "internal2"],
        )
        bundled = obj.process_file(file_path="./test_data/entrypoint.py")
        print(bundled)
        assert bundled is not None
