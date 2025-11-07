"""Bundle a pure-python project into a single file.

Example usage:
python bundler_v2.py \
    --packages toylib toylib_projects \
    --input_path "../tinystories/train.py" \
    --output_path foo.py
"""

from typing import Iterable, Optional, Set, Union
import ast
import collections
import dataclasses
import importlib.util


def reverse_graph(graph):
    # Standard algorithms assume A->B means that A comes before B
    # Reverse the edges
    new_graph = collections.defaultdict(list)
    for module, dependencies in graph.items():
        for dep in dependencies:
            new_graph[dep].append(module)
    return new_graph


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    # Kahn's algorithm
    result = []
    q = collections.deque()
    in_degrees = collections.defaultdict(int)

    # Compute in-degree of each node
    for source in graph:
        for dest in graph[source]:
            in_degrees[dest] += 1

    # Find the starting node(s) with in-degree=0
    for node in graph:
        if in_degrees[node] == 0:
            q.append(node)

    # Peform a BFS from starting nodes
    while q:
        node = q.popleft()
        result.append(node)

        # reduce the in-degree of each neighbor by 1
        for neighbor in graph[node]:
            in_degrees[neighbor] -= 1

            # if in-degree becomes 0, add to queue
            if in_degrees[neighbor] == 0:
                q.append(neighbor)

    # return final sorted order
    if len(result) != len(graph):
        raise ValueError("Graph has at least one cycle")

    return result


@dataclasses.dataclass
class Module:
    name: Optional[str] = None
    path: Optional[str] = None

    package: Optional[str] = None
    tree: Optional[ast.AST] = None

    @property
    def id(self) -> str:
        return self.path


def get_module_location(module_name: str) -> Optional[str]:
    """Get the disk location for a module."""
    try:
        spec = importlib.util.find_spec(module_name)

        if spec is None:
            return None

        if spec.origin:
            return spec.origin
        elif spec.submodule_search_locations:
            return str(spec.submodule_search_locations[0])
        else:
            return None

    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return None


class ImportRemover(ast.NodeTransformer):
    """Remove all import statements, separately return external imports."""

    def __init__(self, bundled_packages: Set[str]):
        self.bundled_packages = bundled_packages
        # Results
        self.external_import_statements = []
        self.imports_to_bundle = []  # type: list[Module]
        self.imported_names = []  # type: list[str]

    def _visit_helper(
        self, imp: Module, node: Union[ast.Import, ast.ImportFrom], imported_name: str
    ) -> None:
        imp.path = get_module_location(imp.name)
        if imp.package in self.bundled_packages:
            self.imports_to_bundle.append(imp)
            self.imported_names.append(imported_name)
        else:
            # Get the import statement as a string
            import_statement = ast.unparse(node)
            self.external_import_statements.append(import_statement)

    def visit_Import(self, node: ast.Import) -> Optional[ast.Import]:
        # Handle: import foo, import foo.bar, import foo.bar as baz
        for alias in node.names:
            full_module = alias.name  # e.g., "foo.bar"
            package = full_module.split(".")[0]  # e.g., "foo"
            imported_name = alias.asname if alias.asname else alias.name

            m = Module(
                package=package,
                name=full_module,
            )

            self._visit_helper(m, node, imported_name)
        return None  # Remove the import statement from the tree

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.ImportFrom]:
        """Remove from-import statements from target packages."""
        # Handle: from bar.baz import boo, from bar.baz import boo as qux
        if node.module:  # Skip relative imports for now
            from_module = node.module  # e.g., "bar.baz"
            package = from_module.split(".")[0]  # e.g., "bar"

            for alias in node.names:
                imported_item = alias.name  # e.g., "boo"
                imported_name = alias.asname if alias.asname else alias.name

                # Full path to the imported item
                full_module_path = f"{from_module}.{imported_item}"

                m = Module(
                    package=package,
                    name=full_module_path,
                )
                self._visit_helper(m, node, imported_name)

        return None  # Remove the import statement from the tree


class AttributeStripper(ast.NodeTransformer):
    """Strip module prefixes from attribute accesses."""

    def __init__(self, imported_names: Set[str]):
        self.imported_names = imported_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Replace attribute chains that start with imported names."""
        # Get the root of the attribute chain and collect all attributes
        root = node
        attrs = [node.attr]

        while isinstance(root.value, ast.Attribute):
            attrs.insert(0, root.value.attr)
            root = root.value

        # Check if the root is a Name node with an imported name
        if isinstance(root.value, ast.Name) and root.value.id in self.imported_names:
            # Replace the entire chain with just the final attribute
            return ast.Name(id=attrs[-1], ctx=ast.Load())

        # If not matching our pattern, continue visiting children normally
        return self.generic_visit(node)


@dataclasses.dataclass
class Bundler:
    """Bundle a pure-python project into a single file.

    Packages all the code from `packages_to_bundle` into a single script,
    removing all import statements for those packages and adjusting attribute
    accesses accordingly. All other packages are treated as external imports
    and are retained at the top of the bundled script.

    Returns:
        str: bundled script content
    """

    packages_to_bundle: Iterable[str]

    def __post_init__(self):
        self.packages_to_bundle = set(self.packages_to_bundle)

        # List of all external imports
        self.external_import_statements = []
        # List of all imported names that need to be stripped from attribute accesses
        self.imported_names = []
        # All parsed modules: path -> Module
        self.parsed_modules = {}
        # Topologically sorted modules (by path)
        self.topologically_sorted_modules = []

    def parse_modules_dfs(self, module: Module) -> str:
        assert module.path

        # Skip if already parsed
        if module.id in self.parsed_modules:
            return

        # Read and process the file
        imports_to_bundle, tree = self.process_file(module.path)
        module.tree = tree

        # Recurse through all the imports that need to be bundled
        for imp in imports_to_bundle:
            self.parse_modules_dfs(imp)

        # All dependencies processed, add to parsed modules
        self.parsed_modules[module.id] = module
        self.topologically_sorted_modules.append(module.id)

    def _section_header(self, section_name: str) -> str:
        return "\n".join(
            [
                "# " + "=" * 60,
                f"# {section_name}",
                "# " + "=" * 60,
            ]
        )

    def __call__(self, path: str) -> str:
        self.parse_modules_dfs(Module(path=path))
        print(self.topologically_sorted_modules)

        final_code = []

        # Add all the external imports
        final_code.append(self._section_header("External Imports"))
        final_code.append("\n".join(sorted(set(self.external_import_statements))))

        # Add the code from the bundled modules in topological order
        for mod_id in self.topologically_sorted_modules:
            module = self.parsed_modules[mod_id]

            final_code.append(self._section_header(f"{module.name} - {module.path}"))
            final_code.append(self.post_process_ast_and_convert_to_source(module.tree))

        return "\n\n".join(final_code)

    def process_file(self, file_path: str):
        # Read file source
        with open(file_path, "r") as f:
            source = f.read()

        # Parse the AST
        tree = ast.parse(source)

        # Remove all imports and classify as bundled or external
        import_remover = ImportRemover(self.packages_to_bundle)
        tree = import_remover.visit(tree)

        # Store the external imports - they're bundled at the end
        self.external_import_statements.extend(
            import_remover.external_import_statements
        )
        self.imported_names.extend(import_remover.imported_names)

        # Return the list of the imports that need to be read
        return (import_remover.imports_to_bundle, tree)

    def post_process_ast_and_convert_to_source(self, tree: ast.AST) -> str:
        # Post-process AST to strip attribute accesses
        stripper = AttributeStripper(set(self.imported_names))
        tree = stripper.visit(tree)
        ast.fix_missing_locations(tree)

        # Convert back to source code
        modified_code = ast.unparse(tree)

        return modified_code


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bundle a pure-python project into a single file."
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        required=True,
        help="Package names to bundle (e.g., mypackage)",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the main entry point module",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for the bundled file",
    )

    args = parser.parse_args()
    print(args.packages)

    # Create bundler and run it
    bundler = Bundler(packages_to_bundle=args.packages)
    bundled_code = bundler(path=args.input_path)

    # Write output
    with open(args.output_path, "w") as f:
        f.write(bundled_code)

    print(f"Successfully bundled code to {args.output_path}")
    print(f"Bundled packages: {', '.join(args.packages)}")
    print(f"Total modules processed: {len(bundler.parsed_modules)}")


if __name__ == "__main__":
    main()
