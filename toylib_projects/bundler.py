#!/usr/bin/env python3
"""
Python Project Bundler
Consolidates multiple Python files from a project into a single executable file.
"""

import os
import ast
import sys
import re
from pathlib import Path
from typing import Set, List, Tuple
import argparse


class PythonBundler:
    def __init__(
        self,
        project_path: str,
        entry_point: str = None,
        local_modules: List[str] = None,
    ):
        self.project_path = Path(project_path)
        self.entry_point = entry_point
        self.external_imports = set()
        self.local_imports = set()
        self.processed_files = set()
        self.file_contents = []
        # Track local module names that need to be removed from code
        self.local_module_names = set()
        # Track import aliases (alias -> real_name)
        self.import_aliases = {}
        # Configurable list of local modules to bundle
        self.configured_local_modules = local_modules or []
        # Track specific submodule files that need to be included
        self.required_submodules = set()

    def is_external_import(self, module_name: str) -> bool:
        """Check if an import is external (not part of the project)"""
        # Check if it's a relative import
        print(f"  Checking if '{module_name}' is external...")
        if module_name.startswith("."):
            print(f"    -> Relative import, considered local")
            return False

        # Check against configured local modules
        for local_module in self.configured_local_modules:
            if module_name == local_module or module_name.startswith(
                f"{local_module}."
            ):
                print(f"    -> Matches configured local module '{local_module}'")
                return False

        # Check if module exists in project directory
        parts = module_name.split(".")
        for i in range(len(parts), 0, -1):
            partial = ".".join(parts[:i])
            module_path = self.project_path / f"{partial.replace('.', '/')}.py"
            module_init = self.project_path / partial.replace(".", "/") / "__init__.py"

            # Also check parent directory for the configured modules
            parent_path = self.project_path.parent
            parent_module_path = parent_path / f"{partial.replace('.', '/')}.py"
            parent_module_init = parent_path / partial.replace(".", "/") / "__init__.py"

            if (
                module_path.exists()
                or module_init.exists()
                or parent_module_path.exists()
                or parent_module_init.exists()
            ):
                print(f"    -> Found local module file for '{partial}'")
                return False

        print(f"    -> Considered external")
        return True

    def extract_imports(self, file_path: Path) -> Tuple[Set[str], List[str], Set[str]]:
        """Extract imports and code from a Python file
        Returns: (external_imports, code_lines, local_module_names)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Warning: Syntax error in {file_path}: {e}")
            return set(), [content], set()

        external_imports = set()
        local_module_names = set()
        lines_to_remove = set()

        # Track which lines contain imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Mark the import line(s) for removal
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno - 1 if hasattr(node, "end_lineno") else start_line
                )
                for line_num in range(start_line, end_line + 1):
                    lines_to_remove.add(line_num)

                # Process Import nodes
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_base = alias.name.split(".")[0]
                        if self.is_external_import(module_base):
                            import_stmt = f"import {alias.name}"
                            if alias.asname:
                                import_stmt += f" as {alias.asname}"
                            external_imports.add(import_stmt)
                        else:
                            # Track local module import
                            if alias.asname:
                                # If there's an alias, track it
                                local_module_names.add(alias.asname)
                                self.import_aliases[alias.asname] = alias.name
                            else:
                                # Track the module name itself
                                local_module_names.add(alias.name)

                # Process ImportFrom nodes
                elif isinstance(node, ast.ImportFrom):
                    # Skip relative imports (they're always local)
                    if node.level > 0:
                        # For relative imports like "from . import module"
                        for n in node.names:
                            if n.name != "*":
                                name_to_track = n.asname if n.asname else n.name
                                local_module_names.add(name_to_track)
                        continue

                    if node.module:
                        module_base = node.module.split(".")[0]
                        if self.is_external_import(module_base):
                            if node.names[0].name == "*":
                                import_stmt = f"from {node.module} import *"
                            else:
                                names = ", ".join(
                                    [
                                        f"{n.name}"
                                        if not n.asname
                                        else f"{n.name} as {n.asname}"
                                        for n in node.names
                                    ]
                                )
                                import_stmt = f"from {node.module} import {names}"
                            external_imports.add(import_stmt)
                        else:
                            # Track local imports from modules
                            for n in node.names:
                                if n.name != "*":
                                    name_to_track = n.asname if n.asname else n.name
                                    local_module_names.add(name_to_track)
                                    if n.asname:
                                        self.import_aliases[n.asname] = (
                                            f"{node.module}.{n.name}"
                                        )

                                    # Track the specific submodule file that needs to be included
                                    # For "from toylib.nn import attention", we need toylib/nn/attention.py
                                    full_module_path = (
                                        f"{node.module}.{n.name}".replace(".", "/")
                                    )
                                    self.required_submodules.add(full_module_path)

        # Remove import lines from code
        code_lines = content.split("\n")
        code_without_imports = []
        for i, line in enumerate(code_lines):
            if i not in lines_to_remove:
                code_without_imports.append(line)

        return external_imports, code_without_imports, local_module_names

    def get_required_submodule_files(self) -> List[Path]:
        """Get file paths for all required submodules"""
        submodule_files = []
        print("Required submodules to include:", self.required_submodules)

        for submodule_path in self.required_submodules:
            print(submodule_path)
            # Try different possible locations for the submodule
            possible_paths = [
                Path(f"{submodule_path}.py"),
            ]

            for path in possible_paths:
                print(path)
                if path.exists() and path not in submodule_files:
                    submodule_files.append(path)
                    print(f"  Found required submodule: {path}")
                    break
            else:
                print(f"  Warning: Could not find required submodule: {submodule_path}")

        return submodule_files

    def get_python_files(self) -> List[Path]:
        """Get all Python files in the project and configured local modules"""
        python_files = []

        # Get files from main project directory
        for root, dirs, files in os.walk(self.project_path):
            # Skip virtual environments and cache directories
            dirs[:] = [
                d
                for d in dirs
                if d not in ["venv", "env", ".venv", "__pycache__", ".git"]
            ]

            for file in files:
                if file.endswith(".py") and file != "__pycache__":
                    python_files.append(Path(root) / file)

        # Get files from configured local modules
        for module_name in self.configured_local_modules:
            # Try to find the module in the parent directory
            module_path = self.project_path.parent / module_name.replace(".", "/")
            if module_path.exists() and module_path.is_dir():
                for root, dirs, files in os.walk(module_path):
                    # Skip virtual environments and cache directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if d not in ["venv", "env", ".venv", "__pycache__", ".git"]
                    ]

                    for file in files:
                        if file.endswith(".py") and file != "__pycache__":
                            python_files.append(Path(root) / file)

            # Also try in the project directory itself
            module_path = self.project_path / module_name.replace(".", "/")
            if module_path.exists() and module_path.is_dir():
                for root, dirs, files in os.walk(module_path):
                    # Skip virtual environments and cache directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if d not in ["venv", "env", ".venv", "__pycache__", ".git"]
                    ]

                    for file in files:
                        if file.endswith(".py") and file != "__pycache__":
                            file_path = Path(root) / file
                            if file_path not in python_files:  # Avoid duplicates
                                python_files.append(file_path)

        return python_files

    def process_file(self, file_path: Path) -> None:
        """Process a single Python file"""
        if file_path in self.processed_files:
            return

        self.processed_files.add(file_path)

        # Determine relative path for display
        try:
            relative_path = file_path.relative_to(self.project_path)
        except ValueError:
            # File is outside project path (from configured modules)
            # Try to get a meaningful relative path
            try:
                relative_path = file_path.relative_to(self.project_path.parent)
            except ValueError:
                relative_path = file_path

        print(f"Processing: {relative_path}")

        imports, code, local_modules = self.extract_imports(file_path)
        self.external_imports.update(imports)
        self.local_module_names.update(local_modules)
        print(f"  Found {len(imports)} external imports")
        print(f"  Found {len(local_modules)} local module imports")
        print(
            f" Local modules: {', '.join(sorted(local_modules))}"
            if local_modules
            else " No local modules"
        )

        # Add file header comment
        self.file_contents.append(f"\n# {'=' * 60}")
        self.file_contents.append(f"# File: {relative_path}")
        self.file_contents.append(f"# {'=' * 60}\n")

        # Add the code
        self.file_contents.extend(code)

    def remove_module_prefixes(self, code: str) -> str:
        """Remove local module prefixes from the code"""
        if not self.local_module_names:
            return code

        # Sort module names by length (longest first) to handle nested modules correctly
        sorted_modules = sorted(self.local_module_names, key=len, reverse=True)

        for module_name in sorted_modules:
            # Create a regex pattern that matches the module name followed by a dot
            # and then a valid Python identifier, but not if it's part of a larger identifier
            # Use word boundary at the start to avoid matching partial names
            pattern = r"\b" + re.escape(module_name) + r"\.(\w+)"

            # Replace module.attribute with just attribute
            code = re.sub(pattern, r"\1", code)

            print(f"  Removed prefix: {module_name}")

        return code

    def bundle_helper(self) -> str:
        """Bundle all Python files into a single file"""
        # Get all Python files
        python_files = self.get_python_files()

        if not python_files:
            raise ValueError(f"No Python files found in {self.project_path}")

        # Sort files to ensure consistent ordering
        python_files.sort()

        # If entry point is specified, process it last
        if self.entry_point:
            entry_file = self.project_path / self.entry_point
            if entry_file in python_files:
                python_files.remove(entry_file)
                python_files.append(entry_file)

        # Process all files (this will collect required submodules)
        for file_path in python_files:
            self.process_file(file_path)

        # Now process any additional required submodule files
        print("\nChecking for required submodules...")
        submodule_files = self.get_required_submodule_files()
        print("Submodule files", submodule_files)
        for file_path in submodule_files:
            if file_path not in self.processed_files:
                self.process_file(file_path)

        # Build the final bundled file
        bundled_content = []

        # Header
        bundled_content.append("#!/usr/bin/env python3")
        bundled_content.append('"""')
        bundled_content.append(f"Bundled Python Project: {self.project_path.name}")
        bundled_content.append(
            "This file contains all project code consolidated into a single file."
        )
        bundled_content.append('"""')
        bundled_content.append("")

        # External imports
        if self.external_imports:
            bundled_content.append("# " + "=" * 60)
            bundled_content.append("# External Imports")
            bundled_content.append("# " + "=" * 60)
            for imp in sorted(self.external_imports):
                bundled_content.append(imp)
            bundled_content.append("")

        # Project code
        bundled_content.append("# " + "=" * 60)
        bundled_content.append("# Project Code")
        bundled_content.append("# " + "=" * 60)
        bundled_content.extend(self.file_contents)

        return "\n".join(bundled_content)

    def post_process(self, bundled_code: str) -> str:
        """Post-processing to remove local module prefixes"""
        print(
            f"\nPost-processing: Removing {len(self.local_module_names)} local module prefixes..."
        )

        # Remove module prefixes from the code
        processed_code = self.remove_module_prefixes(bundled_code)

        return processed_code

    def bundle(self) -> str:
        """Main method to bundle the project"""
        bundled_code = self.bundle_helper()
        return self.post_process(bundled_code)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle a Python project into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python python_bundler.py /path/to/project
  python python_bundler.py /path/to/project -e main.py
  python python_bundler.py . -o bundled_project.py
  python python_bundler.py . -m toylib,toylib_projects
  python python_bundler.py . --local-modules toylib_projects --local-modules external_lib
        """,
    )

    parser.add_argument("project_path", help="Path to the Python project directory")
    parser.add_argument("-e", "--entry-point", help="Entry point file (e.g., main.py)")
    parser.add_argument(
        "-o",
        "--output",
        default="bundled_output.py",
        help="Output file name (default: bundled_output.py)",
    )
    parser.add_argument(
        "-m",
        "--local-modules",
        action="append",
        help="Local module names to bundle (can be specified multiple times, or comma-separated)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.project_path):
        print(f"Error: Project path '{args.project_path}' does not exist")
        sys.exit(1)

    # Process local modules argument
    local_modules = []
    if args.local_modules:
        for module_arg in args.local_modules:
            # Support both comma-separated and multiple -m flags
            local_modules.extend([m.strip() for m in module_arg.split(",")])

    try:
        bundler = PythonBundler(args.project_path, args.entry_point, local_modules)
        bundled_code = bundler.bundle()

        # Write to output file
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(bundled_code)

        print(f"\n✅ Successfully bundled project to: {output_path}")
        print("📊 Statistics:")
        print(f"   - Files processed: {len(bundler.processed_files)}")
        print(f"   - External imports found: {len(bundler.external_imports)}")
        print(f"   - Local module prefixes removed: {len(bundler.local_module_names)}")
        if local_modules:
            print(f"   - Configured local modules: {', '.join(local_modules)}")
        print(f"   - Output file size: {output_path.stat().st_size:,} bytes")
        print(
            f"\n📋 You can now copy the contents of '{output_path}' to your target environment"
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
