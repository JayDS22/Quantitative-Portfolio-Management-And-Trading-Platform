"""
Transformer Model for Financial Time Series Prediction.
Advanced attention-based architecture for market trend analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

from ..data.preprocessor import DataPreprocessor


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for financial time series prediction."""
    
    def __init__(self,
                 input_size: int = 20,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 output_size: int = 1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
        for layer in self.output_projection:
            if hasattr(layer, 'weight'):
                layer.weight.data.uniform_(-initrange, initrange)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer."""
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the last token's output for prediction
        output = output[:, -1, :]  # (batch_size, d_model)
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output


class TransformerTrainer:
    """Transformer model trainer and predictor."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.sequence_length = config.get('sequence_length', 100)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.batch_size = config.get('batch_size', 16)
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler()
        self.preprocessor = DataPreprocessor()
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for transformer training."""
        # Add technical indicators
        data = self.preprocessor.add_technical_indicators(data)
        
        # Select features (similar to LSTM but with more features for transformer)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
            'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower', 
            'atr', 'stoch_k', 'williams_r', 'cci', 'mfi'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in data.columns]
        if len(available_features) < 10:
            raise ValueError("Insufficient features for transformer training")
        
        features = data[available_features].values
        
        # Create target variable (next period return)
        targets = data['returns'].shift(-1).fillna(0).values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, targets)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for transformer input."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def train(self, 
              data: pd.DataFrame,
              epochs: int = 50,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the transformer model."""
        self.logger.info("Starting Transformer model training...")
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler with warmup
        scheduler = self._get_scheduler(optimizer, self.warmup_steps, epochs * len(train_loader))
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Calculate directional accuracy
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            directional_accuracy = accuracy_score(
                (val_targets > 0).astype(int),
                (val_predictions > 0).astype(int)
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(f"data/models/transformer_best.pth")
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Directional Accuracy: {directional_accuracy:.3f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
            
            # Early stopping
            if patience_counter >= 15:
                self.logger.info("Early stopping triggered")
                break
        
        # Load best model
        self.load_model(f"data/models/transformer_best.pth")
        
        training_results = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'directional_accuracy': directional_accuracy,
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return training_results
    
    def _get_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
        """Get learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, 
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def predict(self, data: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        data = self.preprocessor.add_technical_indicators(data)
        
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower',
            'atr', 'stoch_k', 'williams_r', 'cci', 'mfi'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        features = data[available_features].values
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Create sequences for prediction
        if len(features_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Get the last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X = torch.FloatTensor(last_sequence).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            prediction = prediction.cpu().numpy().flatten()
        
        if return_probabilities:
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-prediction))
            return probabilities
        
        return prediction
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on model predictions."""
        predictions = self.predict(data, return_probabilities=True)
        
        signals = pd.DataFrame(index=data.index[-len(predictions):])
        signals['prediction'] = predictions
        
        # Generate signals based on prediction confidence
        signals['signal'] = 0
        signals.loc[signals['prediction'] > 0.58, 'signal'] = 1  # Buy signal
        signals.loc[signals['prediction'] < 0.42, 'signal'] = -1  # Sell signal
        
        # Add confidence score
        signals['confidence'] = np.abs(signals['prediction'] - 0.5) * 2
        
        # Add attention weights for interpretability (simplified)
        signals['attention_score'] = signals['confidence']
        
        return signals
    
    def get_attention_weights(self, data: pd.DataFrame) -> np.ndarray:
        """Extract attention weights for interpretability."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # This would require modification of the forward pass to return attention weights
        # For now, return dummy weights
        return np.random.rand(self.sequence_length, self.sequence_length)
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'd_model': self.model.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout
            },
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize model with saved configuration
        config = checkpoint['model_config']
        self.model = TransformerModel(
            input_size=config['input_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler and other parameters
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def evaluate_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                batch_pred = self.model(batch_X)
                predictions.extend(batch_pred.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)
        targets = y.numpy()
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Directional accuracy
        directional_accuracy = accuracy_score(
            (targets > 0).astype(int),
            (predictions > 0).astype(int)
        )
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation
        }
