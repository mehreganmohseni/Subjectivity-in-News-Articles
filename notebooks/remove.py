import nbformat

# Load the notebook
path = "Zeroshot.ipynb"
nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)

# Check and remove widgets metadata if present
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']
    print("Removed metadata.widgets section.")

# Save the updated notebook
nbformat.write(nb, path)
print("Notebook updated successfully.")
