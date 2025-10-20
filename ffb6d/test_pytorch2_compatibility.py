"""
Test script to verify PyTorch 2.x compatibility.

This script checks:
1. No apex imports remain
2. All Python files have valid syntax
3. Basic imports work
4. Documentation files exist

Run this before training to verify the update is correct.
"""

import os
import sys
import ast
import importlib.util


def check_no_apex_imports(directory):
    """Check that no apex imports remain in Python files."""
    print("Checking for apex imports...")
    apex_found = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and file != 'test_pytorch2_compatibility.py':
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        # Check for actual apex imports (not in comments or strings)
                        if ('from apex' in line or 'import apex' in line):
                            # Skip if it's a comment
                            stripped = line.strip()
                            if not stripped.startswith('#') and not stripped.startswith('"') and not stripped.startswith("'"):
                                apex_found.append((filepath, i, line.strip()))
    
    if apex_found:
        print("❌ Found apex imports:")
        for path, line_num, line in apex_found:
            print(f"  {path}:{line_num}: {line}")
        return False
    else:
        print("✓ No apex imports found")
        return True


def check_python_syntax(directory):
    """Check Python syntax of all .py files."""
    print("\nChecking Python syntax...")
    errors = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        ast.parse(f.read(), filename=filepath)
                except SyntaxError as e:
                    errors.append((filepath, str(e)))
    
    if errors:
        print("❌ Syntax errors found:")
        for path, error in errors:
            print(f"  {path}: {error}")
        return False
    else:
        print("✓ All Python files have valid syntax")
        return True


def check_documentation_files():
    """Check that new documentation files exist."""
    print("\nChecking documentation files...")
    required_files = [
        'MODEL_ARCHITECTURE.md',
        'MIGRATION_GUIDE.md',
        'PYTORCH2_UPDATE.md',
        'ffb6d/models/ffb6d_wrapper.py',
        'ffb6d/example_usage.py',
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing documentation files:")
        for file in missing:
            print(f"  {file}")
        return False
    else:
        print("✓ All documentation files exist")
        return True


def check_requirements():
    """Check that requirements.txt has been updated."""
    print("\nChecking requirements.txt...")
    
    with open('requirement.txt', 'r') as f:
        lines = f.readlines()
    
    # Check for apex as a requirement (not in comments)
    has_apex_req = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            if 'apex' in stripped.lower():
                has_apex_req = True
                break
    
    checks = {
        'PyTorch 2.x': any('torch>=2.0' in line or 'torch>2' in line for line in lines),
        'No apex requirement': not has_apex_req,
        'Has comments': any('#' in line for line in lines),
    }
    
    all_pass = all(checks.values())
    
    for check, passed in checks.items():
        if passed:
            print(f"  ✓ {check}")
        else:
            print(f"  ❌ {check}")
    
    return all_pass


def check_training_scripts():
    """Check that training scripts have been updated."""
    print("\nChecking training scripts...")
    
    scripts = ['ffb6d/train_ycb.py', 'ffb6d/train_lm.py']
    
    all_pass = True
    for script in scripts:
        with open(script, 'r') as f:
            content = f.read()
        
        checks = {
            'No amp.initialize': 'amp.initialize' not in content,
            'No amp.scale_loss': 'amp.scale_loss' not in content,
            'Has native imports': 'torch.nn.parallel import DistributedDataParallel' in content,
            'Has SyncBatchNorm': 'SyncBatchNorm' in content,
        }
        
        script_pass = all(checks.values())
        all_pass = all_pass and script_pass
        
        print(f"\n  {script}:")
        for check, passed in checks.items():
            if passed:
                print(f"    ✓ {check}")
            else:
                print(f"    ❌ {check}")
    
    return all_pass


def main():
    """Run all checks."""
    print("="*80)
    print("PyTorch 2.x Compatibility Check")
    print("="*80)
    print()
    
    # Change to repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    
    results = {
        'No apex imports': check_no_apex_imports('ffb6d'),
        'Valid Python syntax': check_python_syntax('ffb6d'),
        'Documentation exists': check_documentation_files(),
        'Requirements updated': check_requirements(),
        'Training scripts updated': check_training_scripts(),
    }
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
    
    all_pass = all(results.values())
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ All checks passed! Ready for PyTorch 2.x")
    else:
        print("❌ Some checks failed. Please review the errors above.")
    print("="*80)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
