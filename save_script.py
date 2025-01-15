import os


def save_directory_contents(base_dir, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Walk through the directory
        for root, dirs, files in os.walk(base_dir):
            # Filter out unwanted directories
            dirs[:] = [d for d in dirs if d not in ('venv', '__pycache__', 'fine_tuned_checkpoints')]

            for file_name in files:
                # Skip unwanted files
                if file_name == "__init__.py" or file_name.endswith(".ipynb") or file_name.endswith(".md"):
                    continue

                # Construct the relative file path
                relative_path = os.path.relpath(os.path.join(root, file_name), base_dir)

                # Format the output path
                output_path = f"[{relative_path.replace(os.sep, '/')}]"

                # Write the file path to the output file
                outfile.write(output_path + '\n')

                # Read and write the file content
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content + '\n\n')


# Example usage
# Replace 'your_directory_path' with the directory you want to process
# Replace 'output.txt' with the desired output file name
save_directory_contents('/Users/seungjunlee/Desktop/MLProjects/articleGenerator/', 'output.txt')
