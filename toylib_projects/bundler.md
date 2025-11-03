# Python Bundler v2 - Configurable Local Modules

## New Feature: Configurable Local Module Bundling

The bundler now supports specifying which modules should be treated as "local" and bundled into the output file. This is essential when your project depends on other internal modules that aren't in the same directory.

## Usage

### Basic Command
```bash
python bundler.py /path/to/project -m module1,module2
```

### Multiple Module Specification Methods

1. **Comma-separated list**:
```bash
python bundler.py myproject -m toylib,toylib_projects -o bundle.py
```

2. **Multiple -m flags**:
```bash
python bundler.py myproject -m toylib -m toylib_projects -o bundle.py
```

3. **Mix both approaches**:
```bash
python bundler.py myproject -m toylib,utils -m toylib_projects -o bundle.py
```

## How It Works

### 1. Module Detection
When you specify a module with `-m`, the bundler will:
- Treat all imports from that module as local (not external)
- Search for the module in:
  - The project directory
  - The parent directory of the project
- Bundle all Python files from those modules

### 2. Import Processing
For imports like:
```python
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment
```

When `toylib_projects` is specified as a local module:
- These imports are removed from the bundled file
- The imported names (`data`, `decoder_only_model`, `experiment`) are tracked
- All their code is included in the bundle

### 3. Prefix Removal
The bundler automatically removes module prefixes in the code:

**Before bundling:**
```python
dataset = data.Dataset("path/to/data")
model = decoder_only_model.create_model(config)
trainer = experiment.Trainer(model, dataset)
result = trainer.train_step(data.get_batch())
```

**After bundling:**
```python
dataset = Dataset("path/to/data")
model = create_model(config)
trainer = Trainer(model, dataset)
result = trainer.train_step(get_batch())
```

## Example Project Structure

```
workspace/
├── myproject/
│   ├── main.py
│   └── utils.py
├── toylib/
│   ├── __init__.py
│   ├── nn.py
│   └── utils.py
└── toylib_projects/
    ├── __init__.py
    └── tinystories/
        ├── __init__.py
        ├── data.py
        ├── decoder_only_model.py
        └── experiment.py
```

### Bundle Command:
```bash
cd workspace
python bundler.py myproject -e main.py -m toylib,toylib_projects -o bundled.py
```

### Result:
- All code from `myproject/`, `toylib/`, and `toylib_projects/` is bundled
- External imports (like `numpy`, `torch`) are preserved at the top
- Local module imports are removed
- Module prefixes in code are stripped
- Everything runs as a single file

## Key Benefits

1. **Modular Development**: Develop with proper module structure
2. **Clean Bundling**: Get a single executable file for deployment
3. **Automatic Refactoring**: No manual prefix removal needed
4. **Flexible Configuration**: Specify exactly which modules to bundle
5. **External Dependencies**: Keeps external imports intact

## Advanced Features

### Entry Point Specification
```bash
python bundler.py project -e main.py -m lib1,lib2
```
Ensures the entry point file is processed last, preserving `if __name__ == "__main__":` logic.

### Custom Output Location
```bash
python bundler.py project -o /path/to/output.py -m modules
```

### Statistics
The bundler reports:
- Number of files processed
- External imports found
- Local module prefixes removed
- Configured local modules used
- Output file size

## Important Notes

1. **Module Search Path**: The bundler looks for modules in both the project directory and its parent directory
2. **Duplicate Prevention**: Files are only processed once even if found in multiple locations
3. **Import Styles**: Supports all Python import styles:
   - `import module`
   - `from package import module`
   - `import module as alias`
   - `from package import module as alias`
4. **Nested Modules**: Handles nested module structures correctly
5. **Relative Imports**: Automatically detected as local

## Troubleshooting

### Module Not Found
If a module isn't being bundled:
- Check it's specified with `-m`
- Verify it exists in the project or parent directory
- Ensure it has proper Python package structure (`__init__.py`)

### Prefixes Not Removed
If some prefixes remain:
- Check the import statement is being detected
- Verify the module is in the local modules list
- Look for typos in module names

### External Imports Bundled
If external packages are being bundled:
- Don't include them in the `-m` list
- They should remain as import statements at the top