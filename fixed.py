import nbformat

# Load the notebook
nb = nbformat.read("role_lora_pipeline.ipynb", as_version=4)

# Ensure each code cell has 'outputs' and 'id'
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        if 'outputs' not in cell:
            cell['outputs'] = []
        if 'id' not in cell:
            cell['id'] = f"cell-{i}"

# Save the fixed notebook
nbformat.write(nb, "role_lora_pipeline_fixed.ipynb")
