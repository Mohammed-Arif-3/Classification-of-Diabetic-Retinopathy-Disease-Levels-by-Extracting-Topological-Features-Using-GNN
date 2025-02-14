Here's a detailed **README** file for your project:  

---

# **Classification of Diabetic Retinopathy Disease Levels by Extracting Topological Features Using Graph Neural Networks**  

## **Overview**  
Diabetic Retinopathy (DR) is a severe complication of diabetes that can lead to vision loss. This project leverages **Graph Neural Networks (GNNs)** to classify different levels of DR by extracting topological features from retinal images. The proposed system enhances classification accuracy by integrating **Fully Convolutional Networks (FCN), Variational Autoencoders (VAE), and Graph Convolutional Networks (GCN).**  

## **Features**  
- **Automated DR classification** based on retinal fundus images  
- **Feature extraction using Variational Autoencoders (VAE)** for improved representation  
- **Graph Neural Networks (GNNs)** for topological analysis of retinal structures  
- **Training on Kaggle & EyePACS datasets** for robust model performance  
- **User-friendly interface** for efficient diagnosis  

## **Project Structure**  
```
|-- data/                      # Dataset directory
|   |-- train/                  # Training images
|   |-- test/                   # Testing images
|
|-- models/                    # Pre-trained and trained models
|   |-- fcn_model.pth
|   |-- vae_model.pth
|   |-- gnn_model.pth
|
|-- src/                       # Source code
|   |-- preprocessing.py        # Image preprocessing scripts
|   |-- feature_extraction.py   # VAE & GNN feature extraction
|   |-- train.py                # Training script
|   |-- evaluate.py             # Model evaluation
|   |-- interface.py            # GUI implementation
|
|-- results/                   # Output images, confusion matrices, etc.
|-- README.md                  # Project documentation
|-- requirements.txt            # Dependencies
```

## **Installation**  
1. **Clone the Repository**  
   ```sh
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Download & Prepare Dataset**  
   - Download Kaggle’s **Diabetic Retinopathy Dataset** and EyePACS  
   - Extract images and place them in the `data/` directory  

## **Usage**  

### **1. Preprocess Data**  
```sh
python src/preprocessing.py --dataset data/
```

### **2. Train the Model**  
```sh
python src/train.py --epochs 50 --batch_size 32
```

### **3. Evaluate Performance**  
```sh
python src/evaluate.py --model models/gnn_model.pth
```

### **4. Run the GUI Interface**  
```sh
python src/interface.py
```

## **Results**  
- **Achieved 90.34% accuracy** on the EyePACS dataset  
- **High sensitivity (97.54%) & specificity (89.56%)**  
- **Outperforms CNN-based DR classification models**  

## **Future Enhancements**  
- **Integration with real-time retinal imaging devices**  
- **Explainable AI techniques for model interpretability**  
- **Federated learning for privacy-preserving DR detection**  

## **Contributors**  
- **Mohammed Arif**  
- **Mohammed Abdul Mateen**  
- **Gandla Uma Rani**  
- **Mathangi Dinesh Kumar**  

## **License**  
This project is licensed under the **MIT License**.
