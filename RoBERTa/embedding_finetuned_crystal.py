import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split

def analyze_crystal_embeddings(npy_file, data_file, save_dir='embedding_analysis_1112'):

    embeddings = np.load(npy_file)
    df = pd.read_pickle(data_file)
    

    X = df['new_column']
    y = df['Egap']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    

    texts = X_test.values
    

    os.makedirs(save_dir, exist_ok=True)
    

    def extract_crystal_system(text):
        crystal_systems = {"cubic", "tetragonal", "orthorhombic", 
                         "hexagonal", "trigonal", "monoclinic", "triclinic"}
        
        pattern = r"(cubic|tetragonal|orthorhombic|hexagonal|trigonal|monoclinic|triclinic)"
        match = re.search(pattern, text.lower())
        
        if match:
            return match.group(1)
            
        words = text.lower().split()
        for word in words:
            if word in crystal_systems:
                return word
                
        return "unknown"
    

    crystal_systems = [extract_crystal_system(text) for text in texts]

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of texts: {len(texts)}")
    print(f"Number of crystal systems: {len(crystal_systems)}")
    
    if len(texts) != embeddings.shape[0]:
        raise ValueError(f"Embeddings is not matching with amoumt of text, Embeddings: {embeddings.shape[0]}, Texts: {len(texts)}")

    crystal_system_colors = {
        "cubic": "#FF4B4B",      
        "tetragonal": "#4B4BFF", 
        "orthorhombic": "#4BFF4B", 
        "hexagonal": "#FF4BFF",  
        "trigonal": "#FFB84B",   
        "monoclinic": "#8B4513", 
        "triclinic": "#FFB6C1", 
        "unknown": "#808080"     
    }
    

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(embeddings)
    

    df_plot = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'crystal_system': crystal_systems
    })
    

    plt.figure(figsize=(12, 10))
    

    crystal_counts = pd.Series(crystal_systems).value_counts()
    print("\nCrystal System Distribution:")
    print(crystal_counts)

    sns.scatterplot(
        data=df_plot,
        x='tsne_1',
        y='tsne_2',
        hue='crystal_system',
        palette=crystal_system_colors,
        alpha=0.7
    )
    

    plt.legend(title='Crystal System', loc='lower right',fontsize=19,title_fontsize=19)#, bbox_to_anchor=(0.98, 0.98)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('') 
    plt.ylabel('') 

    plt.tight_layout()
    plt.savefig(f'{save_dir}/crystal_system_embeddings_linear.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return crystal_counts

# 使用示例
if __name__ == "__main__":
    npy_file = './embedding_analysis/linear_embeddings.npy'
    data_file = './data/df_aflow_4.pkl'
    
    crystal_counts = analyze_crystal_embeddings(
        npy_file=npy_file,
        data_file=data_file,
        save_dir='embedding_analysis_1112'
    )