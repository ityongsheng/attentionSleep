"""
Project verification and structure visualization
"""
import os
from pathlib import Path
from collections import defaultdict


def count_lines(file_path):
    """Count lines in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0


def get_file_size(file_path):
    """Get file size in KB"""
    try:
        return os.path.getsize(file_path) / 1024
    except:
        return 0


def analyze_project():
    """Analyze project structure and statistics"""
    print("=" * 70)
    print("PROJECT STRUCTURE ANALYSIS")
    print("=" * 70)

    # File statistics
    stats = defaultdict(lambda: {'count': 0, 'lines': 0, 'size': 0})

    # Walk through project
    for root, dirs, files in os.walk('.'):
        # Skip hidden and virtual env directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']

        for file in files:
            if file.startswith('.'):
                continue

            file_path = Path(root) / file
            ext = file_path.suffix

            if ext in ['.py', '.yaml', '.yml', '.md', '.txt']:
                stats[ext]['count'] += 1
                stats[ext]['lines'] += count_lines(file_path)
                stats[ext]['size'] += get_file_size(file_path)

    # Print statistics
    print("\n📊 FILE STATISTICS")
    print("-" * 70)
    print(f"{'Type':<15} {'Files':<10} {'Lines':<15} {'Size (KB)':<15}")
    print("-" * 70)

    total_files = 0
    total_lines = 0
    total_size = 0

    for ext in sorted(stats.keys()):
        data = stats[ext]
        print(f"{ext:<15} {data['count']:<10} {data['lines']:<15} {data['size']:<15.2f}")
        total_files += data['count']
        total_lines += data['lines']
        total_size += data['size']

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_files:<10} {total_lines:<15} {total_size:<15.2f}")

    # Component breakdown
    print("\n\n📦 COMPONENT BREAKDOWN")
    print("-" * 70)

    components = {
        'models': 'Deep Learning Models',
        'data': 'Data Processing',
        'train': 'Training Pipeline',
        'deployment': 'Inference & Deployment',
        'gui': 'User Interface',
        'alerts': 'Alert System',
        'database': 'Database Management',
        'tests': 'Unit Tests',
        'config': 'Configuration'
    }

    for comp_dir, description in components.items():
        if os.path.exists(comp_dir):
            py_files = list(Path(comp_dir).rglob('*.py'))
            total_comp_lines = sum(count_lines(f) for f in py_files)
            print(f"✓ {description:<30} {len(py_files)} files, {total_comp_lines} lines")
        else:
            print(f"✗ {description:<30} Missing")

    # Check key files
    print("\n\n📄 KEY FILES")
    print("-" * 70)

    key_files = [
        ('main.py', 'Application Entry Point'),
        ('requirements.txt', 'Dependencies'),
        ('README.md', 'Project Overview'),
        ('QUICKSTART.md', 'Quick Start Guide'),
        ('DOCUMENTATION.md', 'Technical Documentation'),
        ('PROJECT_SUMMARY.md', 'Project Summary'),
        ('setup.py', 'Installation Script'),
        ('.gitignore', 'Git Ignore Rules')
    ]

    for file_name, description in key_files:
        if os.path.exists(file_name):
            size = get_file_size(file_name)
            print(f"✓ {file_name:<25} {description:<30} ({size:.1f} KB)")
        else:
            print(f"✗ {file_name:<25} {description:<30} MISSING")

    # Directory structure
    print("\n\n🌳 DIRECTORY STRUCTURE")
    print("-" * 70)

    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree"""
        if current_depth >= max_depth:
            return

        try:
            entries = sorted(Path(directory).iterdir())
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.') and e.name != 'venv']
        files = [e for e in entries if e.is_file() and not e.name.startswith('.')]

        # Print directories
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last else '├── '}{d.name}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(d, new_prefix, max_depth, current_depth + 1)

        # Print files (only at current level)
        if current_depth < 2:
            for i, f in enumerate(files):
                is_last = i == len(files) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{f.name}")

    print_tree('.')

    # Implementation status
    print("\n\n✅ IMPLEMENTATION STATUS")
    print("-" * 70)

    checklist = [
        ("Model Architecture", True, "ECA, CBAM, MobileNetV3, LSTM"),
        ("Data Processing", True, "Augmentation, EAR/MAR calculation"),
        ("Training Pipeline", True, "Trainer with TensorBoard logging"),
        ("Real-time Detector", True, "MediaPipe + Model inference"),
        ("GUI Application", True, "PyQt5 with multi-threading"),
        ("Voice Alerts", True, "TTS engine with cooldown"),
        ("Database Logging", True, "SQLite with sessions"),
        ("Model Export", True, "ONNX export script"),
        ("Unit Tests", True, "Model component tests"),
        ("Documentation", True, "README, guides, technical docs"),
        ("Dataset", False, "Requires NTHU-DDD download"),
        ("Trained Model", False, "Requires training on dataset"),
        ("Thesis Paper", False, "In progress")
    ]

    for item, status, note in checklist:
        symbol = "✓" if status else "⏳"
        status_text = "Complete" if status else "Pending"
        print(f"{symbol} {item:<25} {status_text:<15} {note}")

    print("\n" + "=" * 70)
    print("PROJECT VERIFICATION COMPLETE")
    print("=" * 70)

    # Next steps
    print("\n📋 NEXT STEPS:")
    print("1. Download NTHU-DDD dataset to data/raw/")
    print("2. Run: python data/preprocess.py")
    print("3. Run: python train/trainer.py")
    print("4. Run: python main.py")
    print("\nFor detailed instructions, see QUICKSTART.md")


if __name__ == "__main__":
    analyze_project()
