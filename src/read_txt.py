def load_bangla_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Example usage
file_path = "outputs/clean_bangla_corpus1.txt"
bangla_text = load_bangla_text(file_path)

print("🔍 First 500 characters:")
print(bangla_text[:500])
