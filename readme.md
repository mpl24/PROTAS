# PROTAS

## Quick Start Demo
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt


cd demo
sh ./demo/make_synthetic_data.sh
sh ./demo/make_synthetic_model.sh
```

Open jupyter notebook and run protas_demo.ipyb

## Installation
### Prerequisites
Python 3.8+
CUDA-Capable GPU (for training)
R 4.0+ (for tcga pathway analysis and glmm)

## License

This software is released under the **Academic Software License** (© 2025 UCLA).

- **Who can use it:** Academic or nonprofit researchers, for educational or academic research purposes only.
- **Redistribution:** You may share the Software (and derivative works) **only** with other academic or nonprofit researchers and **only** free of charge, and you must include the full license text.
- **No warranty:** The Software is provided *“as is”* with no warranty of any kind.
- **Publication requirement:** Any academic or scholarly publication arising from the use of this Software must include the following acknowledgment:

  > The Software used in this research was created by Mara Pleasure and the BAIR Lab of UCLA. © 2025 UCLA.

- **Commercial use:** Commercial entities must contact [software@tdg.ucla.edu](mailto:software@tdg.ucla.edu) for licensing opportunities.

Please see the [LICENSE](./license.txt) file for the full terms.
