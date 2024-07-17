## Deep learning prediction of diffusion MRI data with microstructure-sensitive loss functions

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supergengchen/Loss-functions.git
   cd micro_loss
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
### Data Preparation
Explain how to prepare the data for training and testing. Include any preprocessing steps if applicable.

1. **Training Data:** Place training data in `data/train` directory. Use `train_subjects.txt` to specify training subjects.
2. **Testing Data:** Place testing data in `data/test` directory. Use `test_subjects.txt` to specify testing subjects.
3. **Validation Data:** Place validation data in `data/val` directory. Use `val_subjects.txt` to specify validation subjects.
4. **Data Format:** Data should be stored as NumPy arrays (`npy` files) for each subject, with specific two dimensions and the last dimension includes for source, target, and feature data.

### Training and Testing

1. Run demo:
   ```bash
   python demo.py (a sample training example for six shells dMRI data)
2. Run main:
   ```bash
   python mian.py (a all procession for six shells dMRI data)
### Contributing

xxx
