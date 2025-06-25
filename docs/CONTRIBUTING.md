# Contributing Guidelines

## 🚀 Welcome Contributors!

We appreciate your interest in improving our brain ventricle and white matter hyperintensity segmentation framework! This guide will help you get started.

## 🔧 Development Setup

### Prerequisites
- Python 3.9+
- Git
- CUDA-capable GPU (recommended)

### Setup Instructions

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Sim-Vent-WMH-Seg.git
cd Sim-Vent-WMH-Seg

# Create development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to verify setup
python -m pytest tests/
```

## 📝 How to Contribute

### 1. Report Issues
- Use [GitHub Issues](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/issues)
- Include clear description, steps to reproduce, and expected behavior
- Add relevant labels (bug, enhancement, documentation)

### 2. Suggest Enhancements
- Open a [GitHub Discussion](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/discussions)
- Describe the proposed feature and its clinical/technical benefits
- Include implementation ideas if possible

### 3. Submit Code Changes

#### Process
1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow coding standards below
3. **Test thoroughly**: Run all tests and add new ones
4. **Commit**: Use clear, descriptive commit messages
5. **Push**: `git push origin feature/your-feature-name`
6. **Pull Request**: Submit with detailed description

#### Pull Request Guidelines
- **Title**: Clear and descriptive
- **Description**: Explain what, why, and how
- **Testing**: Include test results and new test cases
- **Documentation**: Update relevant docs
- **Single Purpose**: One feature/fix per PR

## 💻 Coding Standards

### Code Style
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
pylint src/

# Type checking
mypy src/
```

### Code Quality Guidelines
- **PEP 8 compliance**: Use black and flake8
- **Type hints**: Add type annotations for functions
- **Docstrings**: Use Google style docstrings
- **Variable names**: Clear, descriptive names
- **Comments**: Explain complex logic, not obvious code

### Example Function
```python
def preprocess_flair_image(
    input_path: str, 
    output_dir: str,
    noise_reduction: bool = True
) -> tuple[np.ndarray, dict]:
    """Preprocess FLAIR MRI image for segmentation.
    
    Args:
        input_path: Path to input FLAIR NIfTI file
        output_dir: Directory to save preprocessed output
        noise_reduction: Whether to apply noise reduction filters
        
    Returns:
        Tuple of preprocessed image array and metadata dict
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If image format is invalid
    """
    # Implementation here
    pass
```

## 🧪 Testing

### Test Structure
```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Integration tests for workflows
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test complete workflows
- **Use fixtures**: For sample data and common setups
- **Mock external dependencies**: File I/O, network calls
- **Test edge cases**: Invalid inputs, boundary conditions

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_preprocessing.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

### Types of Documentation
- **Code comments**: Explain complex logic
- **Docstrings**: Document all public functions/classes
- **README updates**: Keep installation/usage current
- **Tutorial updates**: Update usage examples
- **Clinical guides**: Medical interpretation guidelines

### Documentation Standards
- **Clear language**: Avoid jargon, explain technical terms
- **Examples**: Include code examples and expected outputs
- **Medical context**: Explain clinical significance where relevant
- **Keep updated**: Update docs with code changes

## 🎯 Priority Contribution Areas

### High Priority
- **Multi-site validation**: Test on different scanner data
- **Performance optimization**: Speed and memory improvements
- **Clinical deployment tools**: DICOM integration, GUI development
- **Additional baseline methods**: Implement more comparison algorithms

### Medium Priority
- **Documentation improvements**: Tutorials, clinical guides
- **Testing coverage**: Increase test coverage
- **Visualization tools**: Better result visualization functions
- **CI/CD improvements**: Automated testing and deployment

### Research Extensions
- **3D implementation**: Extend to full 3D processing
- **Multi-modal integration**: Incorporate T1, T2 sequences
- **Longitudinal analysis**: Time series processing capabilities
- **Other neurological conditions**: Extend beyond MS

## 🔍 Review Process

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage maintained
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact assessed
- [ ] Clinical accuracy considerations addressed

### Review Timeline
- **Initial response**: Within 2-3 days
- **Full review**: Within 1 week
- **Feedback incorporation**: Collaborative process
- **Final approval**: After all criteria met

## 🌟 Recognition

Contributors will be:
- **Acknowledged**: In README and documentation
- **Credited**: In relevant publications (for significant contributions)
- **Invited**: To join core development team (for ongoing contributors)

## ❓ Questions?

- **Technical questions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/issues)
- **General inquiries**: Contact maintainers through GitHub

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing medical image analysis! 🧠✨**