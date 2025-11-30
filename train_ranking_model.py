import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample

# --- 1. Configuration ---
# Use a high-performing model for semantic tasks
MODEL_NAME = 'all-mpnet-base-v2' 
DATA_PATH = "data/dev_track_a.jsonl"
BATCH_SIZE = 16
NUM_EPOCHS = 4
EVAL_BATCH_SIZE = 32
# Margin for the MarginRankingLoss (Score_Pos - Score_Neg must be > MARGIN)
MARGIN = 0.5 

# --- 2. Data Loading and Transformation ---

class NarrativeTripletDataset(Dataset):
    """
    Creates a Dataset of (Anchor, Positive, Negative) triplets for training.
    """
    def __init__(self, data_path):
        df = pd.read_json(data_path, lines=True)
        self.triplets = []
        
        for _, row in df.iterrows():
            anchor = row['anchor_text']
            text_a = row['text_a']
            text_b = row['text_b']
            is_closer = row['text_a_is_closer']
            
            # The label dictates which is the Positive and which is the Negative sample
            if is_closer:
                positive = text_a
                negative = text_b
            else:
                positive = text_b
                negative = text_a
            
            # Use InputExample format for sentence-transformers
            # This format is typically (Query, Positive, Negative) but we map it to (Anchor, Positive, Negative)
            self.triplets.append(InputExample(texts=[anchor, positive, negative]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

# --- 3. Model Training ---

def train_ranking_model():
    # Load data and setup model
    train_dataset = NarrativeTripletDataset(DATA_PATH)
    model = SentenceTransformer(MODEL_NAME)

    # Wrap the dataset in a DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Initialize the Margin Ranking Loss
    # We use TripletLoss with an internal conversion to MarginRankingLoss 
    # for simplicity with sentence-transformers.
    train_loss = losses.TripletLoss(model=model, triplet_margin=MARGIN)

    # Start training the model
    print(f"Starting training on {len(train_dataset)} triplets for {NUM_EPOCHS} epochs...")
    
    # Configure the training arguments (optimizer, scheduler are set automatically by default)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100, # A small number of warmup steps is typical
        output_path="output/narrative_ranking_model",
        show_progress_bar=True
    )
    
    print("Training complete. Model saved to output/narrative_ranking_model")
    return model

# --- 4. Prediction and Evaluation ---

def predict_accuracy(model, data_path=DATA_PATH):
    """ Evaluates the model accuracy on the dev set (since we don't have a separate test set here) """
    df = pd.read_json(data_path, lines=True)
    correct_predictions = 0
    
    # We'll batch the stories for efficient embedding (Anchor, A, and B)
    texts_to_embed = df[['anchor_text', 'text_a', 'text_b']].values.tolist()
    
    # Generate embeddings in batches
    print(f"Generating embeddings for {len(df)} stories in prediction mode...")
    embeddings = model.encode([text for sublist in texts_to_embed for text in sublist], 
                              batch_size=EVAL_BATCH_SIZE, 
                              convert_to_tensor=True)
    
    # Reshape the embeddings: [N * 3, Embedding_Dim] -> [N, 3, Embedding_Dim]
    embedding_dim = embeddings.shape[1]
    embeddings_reshaped = embeddings.view(-1, 3, embedding_dim)

    # Separate A, B, and Anchor embeddings
    E_anchor = embeddings_reshaped[:, 0]
    E_A = embeddings_reshaped[:, 1]
    E_B = embeddings_reshaped[:, 2]

    # Calculate Cosine Similarity (our similarity function f(S_i, S_j))
    # Cosine similarity is calculated as dot product of L2-normalized vectors
    sim_A = torch.nn.functional.cosine_similarity(E_anchor, E_A)
    sim_B = torch.nn.functional.cosine_similarity(E_anchor, E_B)

    # The prediction is that the story with higher similarity score is closer
    predictions = (sim_A > sim_B).cpu().numpy()
    
    # Compare with ground truth
    ground_truth = df['text_a_is_closer'].values
    accuracy = (predictions == ground_truth).mean()
    
    print(f"\nModel Accuracy on Dev Set: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    # Ensure output directory exists and a GPU is used if available
    import os
    if not os.path.exists("output"):
        os.makedirs("output")
        
    if torch.cuda.is_available():
        print("GPU available. Using CUDA for training.")
    else:
        print("No GPU available. Training on CPU will be slower.")
        
    # 

    trained_model = train_ranking_model()
    predict_accuracy(trained_model)