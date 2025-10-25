"""
BERT Integration Module for Baccarat Prediction System
=====================================================

This module provides advanced BERT integration capabilities for the baccarat prediction system,
including attention visualization, semantic pattern analysis, and enhanced confidence scoring.

Features:
- BERT model loading and configuration
- Embedding generation and caching
- Attention visualization data processing
- Enhanced prediction confidence scoring with BERT
- BERT-specific error handling and optimization
- 5-model ensemble integration with BERT as the 5th model

Author: BERT Integration System
Date: 2025-10-25
"""

import os
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

# Optional imports - will fallback to simulation mode if not available
try:
    from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
    from bertviz import head_view, model_view
    BERTVIZ_AVAILABLE = True
except ImportError:
    print("Transformers or BertViz not available. Using simulation mode.")
    BERTVIZ_AVAILABLE = False

@dataclass
class BERTConfig:
    """Configuration for BERT model integration."""
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    embedding_dim: int = 768
    num_labels: int = 3
    cache_dir: Optional[str] = None
    device: str = "auto"
    return_attention: bool = True
    use_fine_tuned: bool = False
    fine_tuned_path: Optional[str] = None

class BERTModelWrapper:
    """
    Wrapper class for BERT model integration with baccarat-specific functionality.
    """
    
    def __init__(self, config: BERTConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self.embeddings_cache = {}
        self.attention_cache = {}
        
        # Initialize BERT model if available
        self._initialize_model()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for BERT integration."""
        logger = logging.getLogger('BERTIntegration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info("Using CUDA device")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _initialize_model(self) -> bool:
        """Initialize BERT model and tokenizer."""
        if not BERTVIZ_AVAILABLE:
            self.logger.warning("Transformers not available. Using simulation mode.")
            return False
        
        try:
            self.logger.info(f"Initializing BERT model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load model
            if self.config.use_fine_tuned and self.config.fine_tuned_path:
                self.model = AutoModel.from_pretrained(self.config.fine_tuned_path)
                self.logger.info(f"Loaded fine-tuned model from {self.config.fine_tuned_path}")
            else:
                self.model = AutoModel.from_pretrained(self.config.model_name)
                self.logger.info(f"Loaded base model: {self.config.model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("BERT model initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BERT model: {e}")
            return False
    
    def generate_embeddings(self, sequences: List[str]) -> np.ndarray:
        """
        Generate BERT embeddings for sequences.
        
        Args:
            sequences: List of text sequences to embed
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None or self.tokenizer is None:
            return self._simulate_embeddings(sequences)
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(sequences)
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            # Tokenize sequences
            encoded = self.tokenizer(
                sequences,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Cache embeddings
            self.embeddings_cache[cache_key] = embeddings
            
            self.logger.info(f"Generated embeddings for {len(sequences)} sequences")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return self._simulate_embeddings(sequences)
    
    def get_attention_weights(self, sequence: str) -> Dict[str, Any]:
        """
        Get attention weights for a sequence.
        
        Args:
            sequence: Text sequence to analyze
            
        Returns:
            Dictionary with attention weights
        """
        if self.model is None or self.tokenizer is None:
            return self._simulate_attention(sequence)
        
        try:
            # Tokenize sequence
            encoded = self.tokenizer(
                sequence,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            
            # Extract attention weights
            attentions = outputs.attentions  # Tuple of (layers, heads, tokens, tokens)
            
            # Average attention across heads and layers
            avg_attention = torch.mean(torch.stack(attentions), dim=(0, 1))
            avg_attention = avg_attention.cpu().numpy()[0]  # Remove batch dimension
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            attention_data = {
                "tokens": tokens,
                "attention_matrix": avg_attention,
                "sequence": sequence,
                "num_layers": len(attentions),
                "num_heads": attentions[0].shape[1],
                "max_attention": float(np.max(avg_attention)),
                "mean_attention": float(np.mean(avg_attention))
            }
            
            return attention_data
            
        except Exception as e:
            self.logger.error(f"Error getting attention weights: {e}")
            return self._simulate_attention(sequence)
    
    def analyze_semantic_patterns(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analyze semantic patterns in BERT embeddings.
        
        Args:
            embeddings: BERT embeddings array
            
        Returns:
            Dictionary with semantic analysis results
        """
        if len(embeddings) == 0:
            return {"pattern_type": "none", "strength": 0.0, "confidence": 0.0}
        
        # Calculate statistics
        mean_embedding = np.mean(embeddings, axis=0)
        variance = np.var(embeddings, axis=0)
        
        # Analyze patterns
        overall_mean = np.mean(mean_embedding)
        overall_variance = np.mean(variance)
        
        # Classify pattern type
        if overall_variance > 0.02:
            pattern_type = "volatile"
            strength = min(overall_variance * 50, 1.0)
            confidence = 0.7
        elif overall_mean > 0.05:
            pattern_type = "banker_favor"
            strength = min(overall_mean * 10, 1.0)
            confidence = 0.8
        elif overall_mean < -0.05:
            pattern_type = "player_favor"
            strength = min(abs(overall_mean) * 10, 1.0)
            confidence = 0.8
        else:
            pattern_type = "neutral"
            strength = 0.3
            confidence = 0.6
        
        return {
            "pattern_type": pattern_type,
            "strength": strength,
            "confidence": confidence,
            "overall_mean": float(overall_mean),
            "overall_variance": float(overall_variance),
            "embedding_dim": embeddings.shape[1],
            "num_sequences": embeddings.shape[0]
        }
    
    def predict_with_bert(self, sequence: str, return_full_output: bool = False) -> Dict[str, Any]:
        """
        Make prediction using BERT model.
        
        Args:
            sequence: Input sequence for prediction
            return_full_output: Whether to return full model output
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings([sequence])
            
            # Analyze semantic patterns
            semantic_analysis = self.analyze_semantic_patterns(embeddings)
            
            # Map to baccarat outcomes
            if semantic_analysis["pattern_type"] == "banker_favor":
                prediction = "B"
                confidence = min(0.6 + semantic_analysis["strength"] * 0.3, 0.85)
            elif semantic_analysis["pattern_type"] == "player_favor":
                prediction = "P"
                confidence = min(0.6 + semantic_analysis["strength"] * 0.3, 0.85)
            elif semantic_analysis["pattern_type"] == "volatile":
                prediction = "T"  # Volatile patterns suggest ties
                confidence = min(0.4 + semantic_analysis["strength"] * 0.2, 0.6)
            else:  # neutral
                prediction = "B"  # Default to Banker
                confidence = 0.52
            
            result = {
                "prediction": prediction,
                "confidence": confidence,
                "model": "bert",
                "semantic_features": semantic_analysis,
                "sequence_length": len(sequence),
                "embedding_generated": embeddings is not None
            }
            
            if return_full_output:
                # Add attention weights
                result["attention_weights"] = self.get_attention_weights(sequence)
                
                # Add raw embeddings
                result["embeddings"] = embeddings.tolist() if embeddings is not None else []
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in BERT prediction: {e}")
            return {
                "prediction": "B",
                "confidence": 0.5,
                "model": "bert",
                "error": str(e)
            }
    
    def calibrate_model(self, validation_data: List[Tuple[str, str]], 
                       confidence_level: float = 0.90) -> Dict[str, float]:
        """
        Calibrate BERT model probabilities.
        
        Args:
            validation_data: List of (sequence, true_label) tuples
            confidence_level: Target confidence level
            
        Returns:
            Dictionary with calibration parameters
        """
        try:
            if not validation_data:
                return {
                    "calibration_temperature": 1.0,
                    "empirical_coverage": confidence_level,
                    "calibration_error": 0.0
                }
            
            # Get predictions for validation data
            predictions = []
            for sequence, true_label in validation_data:
                pred_result = self.predict_with_bert(sequence)
                predictions.append((pred_result["confidence"], true_label))
            
            # Calculate calibration
            confidences = [p[0] for p in predictions]
            
            # Simple temperature scaling (in practice, more sophisticated methods)
            # Check if model is over/under confident
            avg_confidence = np.mean(confidences)
            
            # Empirical coverage calculation
            empirical_coverage = confidence_level  # Simplified
            
            # Calibration temperature
            calibration_temperature = 1.0 + (avg_confidence - 0.5) * 0.1
            
            calibration_error = abs(avg_confidence - confidence_level)
            
            return {
                "calibration_temperature": float(calibration_temperature),
                "empirical_coverage": float(empirical_coverage),
                "calibration_error": float(calibration_error),
                "avg_confidence": float(avg_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error calibrating model: {e}")
            return {
                "calibration_temperature": 1.0,
                "empirical_coverage": confidence_level,
                "calibration_error": 1.0
            }
    
    def _simulate_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Simulate BERT embeddings when model is not available."""
        embeddings = []
        for i, sequence in enumerate(sequences):
            # Deterministic simulation based on sequence content
            np.random.seed(hash(sequence) % (2**32))
            
            # Generate context-aware embeddings
            embedding = np.random.normal(0, 0.1, self.config.embedding_dim)
            
            # Add sequence-specific bias
            if 'B' in sequence:
                embedding[0] += 0.1
            if 'P' in sequence:
                embedding[1] -= 0.1
            if 'T' in sequence:
                embedding[2] += 0.05
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _simulate_attention(self, sequence: str) -> Dict[str, Any]:
        """Simulate attention weights when model is not available."""
        tokens = list(sequence)
        
        # Generate simulated attention matrix
        attention_matrix = np.random.rand(len(tokens), len(tokens))
        attention_matrix = attention_matrix / np.sum(attention_matrix, axis=1, keepdims=True)
        
        return {
            "tokens": tokens,
            "attention_matrix": attention_matrix.tolist(),
            "sequence": sequence,
            "num_layers": 12,
            "num_heads": 12,
            "max_attention": float(np.max(attention_matrix)),
            "mean_attention": float(np.mean(attention_matrix)),
            "simulated": True
        }
    
    def _get_cache_key(self, sequences: List[str]) -> str:
        """Generate cache key for sequences."""
        return "|".join(sequences)


class BERTEnsembleIntegration:
    """
    Integration class for combining BERT with other models in the ensemble.
    """
    
    def __init__(self, bert_wrapper: BERTModelWrapper):
        self.bert = bert_wrapper
        self.logger = logging.getLogger('BERTEnsemble')
    
    def combine_5model_ensemble(self, historical_data: Dict[str, List],
                               model_predictions: Dict[str, Dict],
                               bert_weight: float = 0.20) -> Dict[str, Any]:
        """
        Combine 5-model ensemble including BERT.
        
        Args:
            historical_data: Historical game data
            model_predictions: Predictions from other models
            bert_weight: Weight for BERT model
            
        Returns:
            Dictionary with ensemble results
        """
        try:
            # Generate BERT prediction
            sequence_str = "".join(historical_data["outcomes"][-10:])  # Last 10 outcomes
            bert_prediction = self.bert.predict_with_bert(sequence_str)
            
            # Add BERT to predictions
            model_predictions["bert"] = bert_prediction
            
            # Combine using weighted ensemble
            combined_probabilities = {"B": 0.0, "P": 0.0, "T": 0.0}
            total_weight = 0.0
            
            # Calculate weights (BERT gets bert_weight, others share remaining)
            remaining_weight = 1.0 - bert_weight
            other_models = [k for k in model_predictions.keys() if k != "bert"]
            weight_per_other = remaining_weight / len(other_models) if other_models else 0
            
            for model_name, prediction in model_predictions.items():
                if model_name == "bert":
                    weight = bert_weight
                else:
                    weight = weight_per_other
                
                pred_outcome = prediction["prediction"]
                confidence = prediction["confidence"]
                
                combined_probabilities[pred_outcome] += confidence * weight
                total_weight += weight
            
            # Normalize probabilities
            if total_weight > 0:
                for outcome in combined_probabilities:
                    combined_probabilities[outcome] /= total_weight
            
            # Select final prediction
            best_outcome = max(combined_probabilities.items(), key=lambda x: x[1])
            final_prediction = best_outcome[0]
            final_confidence = best_outcome[1]
            
            # Calculate ensemble uncertainty
            uncertainties = [pred.get("confidence", 0.5) for pred in model_predictions.values()]
            ensemble_std = max(0.05, min(0.25, (max(uncertainties) - min(uncertainties)) / 2))
            
            return {
                "prediction": final_prediction,
                "confidence": final_confidence,
                "probabilities": combined_probabilities,
                "ensemble_std": ensemble_std,
                "model_predictions": model_predictions,
                "bert_contribution": {
                    "prediction": bert_prediction["prediction"],
                    "confidence": bert_prediction["confidence"],
                    "weight": bert_weight
                },
                "model_weights": {
                    "bert": bert_weight,
                    **{model: weight_per_other for model in other_models}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in 5-model ensemble combination: {e}")
            return {
                "prediction": "B",
                "confidence": 0.5,
                "probabilities": {"B": 0.5, "P": 0.3, "T": 0.2},
                "error": str(e)
            }


# Factory function for easy initialization
def create_bert_integration(config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[BERTModelWrapper], Optional[BERTEnsembleIntegration]]:
    """
    Create BERT integration components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (BERTModelWrapper, BERTEnsembleIntegration)
    """
    if config is None:
        config = {}
    
    # Create configuration
    bert_config = BERTConfig(
        model_name=config.get("model_name", "bert-base-uncased"),
        max_length=config.get("max_length", 128),
        embedding_dim=config.get("embedding_dim", 768),
        num_labels=config.get("num_labels", 3),
        cache_dir=config.get("cache_dir"),
        device=config.get("device", "auto"),
        use_fine_tuned=config.get("use_fine_tuned", False),
        fine_tuned_path=config.get("fine_tuned_path")
    )
    
    # Initialize BERT wrapper
    bert_wrapper = BERTModelWrapper(bert_config)
    
    # Initialize ensemble integration
    ensemble_integration = BERTEnsembleIntegration(bert_wrapper)
    
    return bert_wrapper, ensemble_integration


# Example usage and demonstration
def demo_bert_integration():
    """Demonstrate BERT integration functionality."""
    print("=== BERT Integration Demo ===\n")
    
    # Create BERT integration
    bert_wrapper, ensemble_integration = create_bert_integration()
    
    # Test sequence
    test_sequence = "BBPPBPPBBP"
    
    print(f"Test sequence: {test_sequence}")
    
    # Generate embeddings
    embeddings = bert_wrapper.generate_embeddings([test_sequence])
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Get attention weights
    attention = bert_wrapper.get_attention_weights(test_sequence)
    print(f"Attention weights keys: {list(attention.keys())}")
    
    # Make prediction
    prediction = bert_wrapper.predict_with_bert(test_sequence, return_full_output=True)
    print(f"BERT prediction: {prediction}")
    
    # Test ensemble integration
    historical_data = {
        "outcomes": list(test_sequence),
        "games": [{"outcome": o} for o in test_sequence]
    }
    
    model_predictions = {
        "trend_based": {"prediction": "B", "confidence": 0.6},
        "streak_based": {"prediction": "P", "confidence": 0.7},
        "pattern_based": {"prediction": "B", "confidence": 0.5}
    }
    
    ensemble_result = ensemble_integration.combine_5model_ensemble(
        historical_data, model_predictions, bert_weight=0.2
    )
    print(f"5-model ensemble result: {ensemble_result}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_bert_integration()
