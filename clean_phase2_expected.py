# clean_phase2_expected.py
import re
from pathlib import Path

def remove_expected_outputs(input_file: str, output_file: str = None):
    """
    Removes only '#### Expected Output:' sections (with fenced code blocks)
    from the given Markdown file, while leaving other code blocks intact.
    """
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path

    text = input_path.read_text(encoding="utf-8")

    # Regex pattern: match "#### Expected Output:" + code block (``` ... ```)
    pattern = re.compile(
        r"#### Expected Output:\s*```[\s\S]*?```", 
        flags=re.MULTILINE
    )

    cleaned_text = re.sub(pattern, "", text)

    output_path.write_text(cleaned_text, encoding="utf-8")
    print(f"Cleaned file saved to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # python clean_phase2_expected.py "J:/Gold_FX/Phase 2.md"
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clean_phase2_expected.py <phase2.md> [<output.md>]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        remove_expected_outputs(input_file, output_file)
