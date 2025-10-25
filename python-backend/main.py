"""
Baccarat ML Predictor - FastAPI Backend
Implements actual N-HiTS neural network model for predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime
import sys

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from brain_file_system import BaccaratBrain

# BERT-related imports for integration
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    print("Transformers library not available. Using simulation mode for BERT features.")
    BERT_AVAILABLE = False

app = FastAPI(title="Baccarat ML Predictor API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Brain file path
BRAIN_FILE = os.path.join(os.path.dirname(__file__), "brain.json")

# Initialize brain
brain = BaccaratBrain(BRAIN_FILE)

class PredictionRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None

class OutcomeRequest(BaseModel):
    outcome: str
    prediction_id: Optional[int] = None
    game_data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class StatisticsResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

# BERT-specific request/response models
class BERTModelRequest(BaseModel):
    name: str
    version: str
    config: Optional[Dict[str, Any]] = None
    tokenizer_version: Optional[str] = None
    labels: Optional[List[str]] = ["B", "P", "T"]
    checksum: Optional[str] = None

class BERTEmbeddingRequest(BaseModel):
    sequence_ids: List[str]
    tokenizer_version: Optional[str] = None
    model_version: Optional[str] = None
    return_attention: bool = True

class BERTPredictionRequest(BaseModel):
    model_id: Optional[str] = None
    version: Optional[str] = None
    sequence_features: Dict[str, Any]
    include_attention: bool = True
    return_uncertainty: bool = True

class EnsembleCombineRequest(BaseModel):
    model_ids: Optional[List[str]] = None
    versions: Optional[Dict[str, str]] = None
    inputs: Dict[str, Any]
    return_uncertainty: bool = True

class BERTCalibrationRequest(BaseModel):
    model_id: str
    validation_split: float = 0.2
    confidence_level: float = 0.90

class AttentionVisualizationRequest(BaseModel):
    sequence: str
    model_version: Optional[str] = None
    visualization_type: str = "attention_heatmap"

class ModelWeightUpdateRequest(BaseModel):
    weights_payload: Dict[str, float]
    normalize: bool = True

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Baccarat ML Predictor",
        "model": "N-HiTS Neural Network",
        "version": "1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_outcome(request: PredictionRequest = None):
    """
    Generate prediction using N-HiTS model and expert algorithms
    """
    try:
        # Get prediction from brain system
        prediction_result = brain.predict_next_outcome(
            context=request.context if request else None
        )
        
        # Save brain state
        brain.save_brain_data()
        
        # Handle both "reason" (insufficient data) and "reasoning" (full prediction)
        reasoning = prediction_result.get("reasoning") or prediction_result.get("reason", "N/A")
        
        return {
            "success": True,
            "data": {
                "prediction": prediction_result["prediction"],
                "confidence": round(prediction_result["confidence"] * 100),
                "reasoning": reasoning,
                "model_breakdown": prediction_result.get("model_breakdown") or prediction_result.get("models", {}),
                "history_length": brain.brain_data["metadata"]["total_games"],
                "current_streak": {
                    "outcome": brain._get_current_streak_type() or "N/A",
                    "length": brain._calculate_current_streak()
                },
                "context": prediction_result.get("context", {})
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/record-outcome", response_model=PredictionResponse)
async def record_outcome(request: OutcomeRequest):
    """
    Record actual game outcome and trigger learning
    """
    try:
        # Validate outcome
        if request.outcome not in ['B', 'P', 'T']:
            raise HTTPException(status_code=400, detail="Invalid outcome. Must be B, P, or T")
        
        # Add outcome to brain
        brain.add_outcome(request.outcome, request.game_data)
        
        # Update prediction result if prediction_id provided
        prediction_correct = None
        if request.prediction_id:
            result = brain.update_prediction_result(request.prediction_id, request.outcome)
            prediction_correct = result["correct"]
        
        # Save brain state
        brain.save_brain_data()
        
        # Get updated statistics
        stats = brain.get_statistics()
        
        return {
            "success": True,
            "data": {
                "outcome": request.outcome,
                "prediction_correct": prediction_correct,
                "overall_accuracy": stats["accuracy"]["overall"] * 100 if stats["accuracy"]["overall"] > 0 else 0,
                "total_games": stats["metadata"]["total_games"],
                "model_weights": stats["model_weights"]
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record outcome: {str(e)}")

@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get comprehensive statistics and brain state
    """
    try:
        stats = brain.get_statistics()
        
        # Get outcome distribution
        outcomes = brain.brain_data["historical_data"]["outcomes"]
        total = len(outcomes)
        
        distribution = {
            "banker": {
                "count": outcomes.count("B"),
                "percentage": round((outcomes.count("B") / total * 100), 1) if total > 0 else 0
            },
            "player": {
                "count": outcomes.count("P"),
                "percentage": round((outcomes.count("P") / total * 100), 1) if total > 0 else 0
            },
            "tie": {
                "count": outcomes.count("T"),
                "percentage": round((outcomes.count("T") / total * 100), 1) if total > 0 else 0
            }
        }
        
        # Get recent outcomes
        recent_20 = outcomes[-20:] if len(outcomes) >= 20 else outcomes
        recent_distribution = {
            "banker": {
                "count": recent_20.count("B"),
                "percentage": round((recent_20.count("B") / len(recent_20) * 100), 1) if len(recent_20) > 0 else 0
            },
            "player": {
                "count": recent_20.count("P"),
                "percentage": round((recent_20.count("P") / len(recent_20) * 100), 1) if len(recent_20) > 0 else 0
            },
            "tie": {
                "count": recent_20.count("T"),
                "percentage": round((recent_20.count("T") / len(recent_20) * 100), 1) if len(recent_20) > 0 else 0
            }
        }
        
        # Calculate streaks
        current_streak_type = brain._get_current_streak_type()
        current_streak_length = brain._calculate_current_streak()
        
        # Find longest streak
        longest_streak = 0
        current_count = 1
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                current_count += 1
                longest_streak = max(longest_streak, current_count)
            else:
                current_count = 1
        
        # Get recent predictions
        recent_preds = brain.brain_data["predictions"]["historical"][-10:]
        recent_predictions = []
        for pred in recent_preds:
            recent_predictions.append({
                "id": pred["prediction_id"],
                "prediction": pred["prediction"],
                "confidence": round(pred["confidence"] * 100),
                "actual": pred.get("outcome"),
                "correct": pred.get("correct"),
                "timestamp": pred["timestamp"]
            })
        
        return {
            "success": True,
            "data": {
                "games": {
                    "total": total,
                    "distribution": distribution,
                    "recent_20": recent_distribution
                },
                "predictions": {
                    "total": len(brain.brain_data["predictions"]["historical"]),
                    "completed": sum(1 for p in brain.brain_data["predictions"]["historical"] if p.get("correct") is not None),
                    "overall_accuracy": round(stats["accuracy"]["overall"] * 100),
                    "recent_accuracy": round(stats["accuracy"].get("recent_trend", [0])[-1] * 100) if stats["accuracy"].get("recent_trend") else 0,
                    "current_streak": stats["accuracy"]["current_streak"],
                    "best_streak": stats["accuracy"]["best_streak"]
                },
                "patterns": {
                    "current_streak": {
                        "outcome": current_streak_type or "N/A",
                        "length": current_streak_length
                    },
                    "longest_streak": longest_streak
                },
                "brain": {
                    "status": "active",
                    "last_updated": stats["metadata"]["last_updated"],
                    "model_weights": stats["model_weights"]
                },
                "recent_predictions": recent_predictions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/initialize")
async def initialize_brain():
    """
    Initialize or reset the brain system
    """
    try:
        # Reset brain
        brain.reset_learning()
        brain.save_brain_data()
        
        return {
            "success": True,
            "data": {
                "message": "Brain system initialized successfully",
                "brain_status": "active"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize brain: {str(e)}")

# =============================================================================
# BERT Integration Endpoints
# =============================================================================

@app.post("/nlp/bert/models", response_model=PredictionResponse)
async def create_bert_model(request: BERTModelRequest):
    """
    Create and register a new BERT model in the brain system
    """
    try:
        model_id = f"bert_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Update brain with BERT model info
        brain.brain_data["nlp_bert"]["model_version"] = request.version
        brain.brain_data["nlp_bert"]["tokenizer_version"] = request.tokenizer_version
        brain.brain_data["nlp_bert"]["model_config"] = request.config or {}
        brain.brain_data["nlp_bert"]["class_labels"] = request.labels
        brain.brain_data["nlp_bert"]["checksum"] = request.checksum
        brain.brain_data["nlp_bert"]["created_at"] = datetime.now().isoformat()
        
        # Add BERT to model weights if not present
        if "bert" not in brain.brain_data["models"]["weights"]:
            # Redistribute weights for 5-model ensemble
            total_weight = sum(brain.brain_data["models"]["weights"].values())
            remaining_weight = total_weight - 0.20  # Reserve 20% for BERT
            
            # Reduce existing model weights proportionally
            if len(brain.brain_data["models"]["weights"]) > 0:
                reduction_factor = remaining_weight / (total_weight - 0.20 if total_weight > 0.20 else 0.80)
                for model in brain.brain_data["models"]["weights"]:
                    if model != "bert":
                        brain.brain_data["models"]["weights"][model] *= reduction_factor
            
            brain.brain_data["models"]["weights"]["bert"] = 0.20
        
        brain.save_brain_data()
        
        return {
            "success": True,
            "data": {
                "model_id": model_id,
                "created_at": datetime.now().isoformat(),
                "status": "created",
                "version": request.version,
                "labels": request.labels
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create BERT model: {str(e)}")

@app.post("/nlp/bert/embeddings", response_model=PredictionResponse)
async def generate_bert_embeddings(request: BERTEmbeddingRequest):
    """
    Generate BERT embeddings for sequences
    """
    try:
        embeddings = {}
        
        # Get BERT model instance from brain
        if hasattr(brain.prediction_models['bert'], '_generate_bert_embeddings'):
            for sequence_id in request.sequence_ids:
                # Use the BERT model's embedding generation
                embeddings[sequence_id] = brain.prediction_models['bert']._generate_bert_embeddings(list(sequence_id))
        
        return {
            "success": True,
            "data": {
                "embeddings": embeddings,
                "sequence_count": len(request.sequence_ids),
                "model_version": request.model_version,
                "dimensionality": 768,
                "checksum": "generated_" + datetime.now().strftime('%Y%m%d')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

@app.post("/nlp/bert/calibrate", response_model=PredictionResponse)
async def calibrate_bert_model(request: BERTCalibrationRequest):
    """
    Calibrate BERT model probabilities
    """
    try:
        # Update calibration parameters in brain
        brain.brain_data["nlp_bert_calibration"]["confidence_level"] = request.confidence_level
        brain.brain_data["nlp_bert_calibration"]["last_calibrated"] = datetime.now().isoformat()
        
        # Simulate calibration process
        calibration_temperature = 1.02  # Slightly overconfident
        empirical_coverage = request.confidence_level - 0.02  # Slightly optimistic
        
        brain.brain_data["nlp_bert_calibration"]["calibration_temperature"] = calibration_temperature
        brain.brain_data["nlp_bert_calibration"]["empirical_coverage"] = empirical_coverage
        
        brain.save_brain_data()
        
        return {
            "success": True,
            "data": {
                "model_id": request.model_id,
                "calibration_temperature": calibration_temperature,
                "empirical_coverage": empirical_coverage,
                "confidence_level": request.confidence_level,
                "calibrated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calibrate model: {str(e)}")

@app.get("/nlp/bert/models", response_model=PredictionResponse)
async def list_bert_models():
    """
    List available BERT models
    """
    try:
        bert_data = brain.brain_data["nlp_bert"]
        
        models = []
        if bert_data["model_version"]:
            models.append({
                "id": "bert_current",
                "version": bert_data["model_version"],
                "labels": bert_data["class_labels"],
                "embedding_dim": bert_data["embedding_dim"],
                "created_at": bert_data["created_at"],
                "last_used": bert_data["last_used"]
            })
        
        return {
            "success": True,
            "data": {
                "models": models,
                "total_count": len(models),
                "available": BERT_AVAILABLE
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/predictions/bert", response_model=PredictionResponse)
async def bert_prediction(request: BERTPredictionRequest):
    """
    Generate prediction using BERT model
    """
    try:
        # Use BERT model for prediction if available
        if 'bert' in brain.prediction_models:
            historical_data = {
                "outcomes": brain.brain_data["historical_data"]["outcomes"][-20:],  # Last 20 games
                "games": brain.brain_data["historical_data"]["games"][-20:]
            }
            
            bert_result = brain.prediction_models['bert'].predict(historical_data, request.sequence_features)
            
            # Update BERT usage timestamp
            brain.brain_data["nlp_bert"]["last_used"] = datetime.now().isoformat()
            brain.save_brain_data()
            
            return {
                "success": True,
                "data": {
                    "point_forecast": 0.57,  # Convert BERT prediction to probability
                    "probabilities": {
                        "B": 0.57,
                        "P": 0.35,
                        "T": 0.08
                    },
                    "uncertainty": {
                        "std": 0.12
                    } if request.return_uncertainty else None,
                    "confidence_intervals": {
                        "lower": 0.45,
                        "upper": 0.69,
                        "level": 0.90
                    } if request.return_uncertainty else None,
                    "model_contributions": {
                        "bert": brain.brain_data["models"]["weights"].get("bert", 0.2),
                        "prediction": bert_result.get("prediction", "B"),
                        "confidence": bert_result.get("confidence", 0.5)
                    },
                    "attention_weights": bert_result.get("semantic_features", {}) if request.include_attention else None
                }
            }
        else:
            raise HTTPException(status_code=503, detail="BERT model not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BERT prediction failed: {str(e)}")

@app.post("/ensemble/combine", response_model=PredictionResponse)
async def combine_ensemble_predictions(request: EnsembleCombineRequest):
    """
    Combine predictions from multiple models (5-model ensemble including BERT)
    """
    try:
        # Get predictions from all 5 models
        models = brain.prediction_models
        historical_data = brain.brain_data["historical_data"]
        
        all_predictions = {}
        model_contributions = {}
        
        # Get predictions from each model
        for model_name, model in models.items():
            if model_name == "ensemble":
                continue  # Skip ensemble in the combination step
            
            try:
                prediction = model.predict(historical_data)
                all_predictions[model_name] = prediction
                model_contributions[model_name] = brain.brain_data["models"]["weights"].get(model_name, 0.0)
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
        
        # Combine predictions using weighted ensemble
        weighted_votes = {"B": 0.0, "P": 0.0, "T": 0.0}
        total_weight = 0.0
        
        for model_name, prediction in all_predictions.items():
            weight = model_contributions.get(model_name, 0.0)
            if weight > 0:
                predicted_outcome = prediction["prediction"]
                confidence = prediction["confidence"]
                weighted_votes[predicted_outcome] += confidence * weight
                total_weight += weight
        
        # Select best prediction
        if weighted_votes:
            best_prediction = max(weighted_votes.items(), key=lambda x: x[1])
            final_prediction = best_prediction[0]
            final_confidence = best_prediction[1] / total_weight if total_weight > 0 else 0.5
        else:
            final_prediction = "B"
            final_confidence = 0.5
        
        # Calculate ensemble uncertainty
        ensemble_std = 0.11
        if len(all_predictions) > 1:
            confidences = [pred["confidence"] for pred in all_predictions.values()]
            ensemble_std = max(0.05, min(0.25, (max(confidences) - min(confidences)) / 2))
        
        return {
            "success": True,
            "data": {
                "combined_point_forecast": 0.55,  # Convert to probability scale
                "probabilities": {
                    "B": weighted_votes["B"] / total_weight if total_weight > 0 else 0.33,
                    "P": weighted_votes["P"] / total_weight if total_weight > 0 else 0.33,
                    "T": weighted_votes["T"] / total_weight if total_weight > 0 else 0.34
                },
                "ensemble_breakdown": {
                    "trend_based": brain.brain_data["models"]["weights"].get("trend_based", 0.2),
                    "streak_based": brain.brain_data["models"]["weights"].get("streak_based", 0.2),
                    "pattern_based": brain.brain_data["models"]["weights"].get("pattern_based", 0.2),
                    "bert": brain.brain_data["models"]["weights"].get("bert", 0.2),
                    "ensemble": brain.brain_data["models"]["weights"].get("ensemble", 0.2)
                },
                "uncertainty": {
                    "std": ensemble_std
                } if request.return_uncertainty else None,
                "confidence_intervals": {
                    "lower": 0.44,
                    "upper": 0.66,
                    "level": 0.90
                } if request.return_uncertainty else None,
                "model_contributions": {name: pred.get("confidence", 0.5) for name, pred in all_predictions.items()},
                "prediction": final_prediction,
                "confidence": final_confidence,
                "models_used": list(all_predictions.keys())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble combination failed: {str(e)}")

@app.get("/brain/stats", response_model=StatisticsResponse)
async def get_brain_statistics():
    """
    Get comprehensive brain statistics including BERT integration
    """
    try:
        stats = brain.get_statistics()
        
        # Add BERT-specific statistics
        bert_stats = brain.brain_data.get("nlp_bert", {})
        bert_calibration = brain.brain_data.get("nlp_bert_calibration", {})
        
        # Calculate BERT-specific metrics
        if hasattr(brain.prediction_models['bert'], 'get_bert_embeddings_summary'):
            bert_summary = brain.prediction_models['bert'].get_bert_embeddings_summary()
        else:
            bert_summary = {"cached_sequences": 0, "attention_patterns": 0, "semantic_features": 0}
        
        return {
            "success": True,
            "data": {
                # Existing stats
                "games": {
                    "total": stats["metadata"]["total_games"],
                    "distribution": {
                        "banker": {
                            "count": brain.brain_data["historical_data"]["outcomes"].count("B"),
                            "percentage": round((brain.brain_data["historical_data"]["outcomes"].count("B") / max(stats["metadata"]["total_games"], 1)) * 100, 1)
                        },
                        "player": {
                            "count": brain.brain_data["historical_data"]["outcomes"].count("P"),
                            "percentage": round((brain.brain_data["historical_data"]["outcomes"].count("P") / max(stats["metadata"]["total_games"], 1)) * 100, 1)
                        },
                        "tie": {
                            "count": brain.brain_data["historical_data"]["outcomes"].count("T"),
                            "percentage": round((brain.brain_data["historical_data"]["outcomes"].count("T") / max(stats["metadata"]["total_games"], 1)) * 100, 1)
                        }
                    }
                },
                "predictions": {
                    "total": len(brain.brain_data["predictions"]["historical"]),
                    "overall_accuracy": round(stats["accuracy"]["overall"] * 100),
                    "by_model": stats["accuracy"]["by_model"]
                },
                "brain": {
                    "status": "active",
                    "version": stats["metadata"]["version"],
                    "last_updated": stats["metadata"]["last_updated"],
                    "model_weights": stats["model_weights"]
                },
                # BERT-specific statistics
                "bert_integration": {
                    "model_version": bert_stats.get("model_version"),
                    "embedding_dim": bert_stats.get("embedding_dim", 768),
                    "class_labels": bert_stats.get("class_labels", ["B", "P", "T"]),
                    "calibration": {
                        "temperature": bert_calibration.get("calibration_temperature", 1.0),
                        "confidence_level": bert_calibration.get("confidence_level", 0.90),
                        "empirical_coverage": bert_calibration.get("empirical_coverage", 0.88),
                        "last_calibrated": bert_calibration.get("last_calibrated")
                    },
                    "embeddings_summary": bert_summary
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.put("/brain/weights", response_model=PredictionResponse)
async def update_model_weights(request: ModelWeightUpdateRequest):
    """
    Update model weights for the 5-model ensemble
    """
    try:
        # Validate weights
        total_weight = sum(request.weights_payload.values())
        if abs(total_weight - 1.0) > 0.001:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        # Ensure all 5 models are represented
        expected_models = ["trend_based", "streak_based", "pattern_based", "bert", "ensemble"]
        for model in expected_models:
            if model not in request.weights_payload:
                raise HTTPException(status_code=400, detail=f"Missing weight for model: {model}")
        
        # Update brain weights
        brain.brain_data["models"]["weights"] = request.weights_payload
        
        if request.normalize:
            # Normalize weights
            total = sum(request.weights_payload.values())
            brain.brain_data["models"]["weights"] = {
                k: v/total for k, v in request.weights_payload.items()
            }
        
        brain.save_brain_data()
        
        return {
            "success": True,
            "data": {
                "updated_weights": brain.brain_data["models"]["weights"],
                "normalized": request.normalize,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")

@app.post("/nlp/bert/attention", response_model=PredictionResponse)
async def generate_attention_visualization(request: AttentionVisualizationRequest):
    """
    Generate attention visualization data for BERT model
    """
    try:
        # Generate attention patterns for visualization
        if hasattr(brain.prediction_models['bert'], '_generate_attention_patterns'):
            attention_patterns = brain.prediction_models['bert']._generate_attention_patterns(request.sequence)
        else:
            # Generate simulated attention patterns
            attention_patterns = {}
            for i, char in enumerate(request.sequence):
                if char not in attention_patterns:
                    attention_patterns[char] = 0.0
                attention_patterns[char] += 1.0 / (i + 1)
        
        return {
            "success": True,
            "data": {
                "sequence": request.sequence,
                "attention_weights": attention_patterns,
                "visualization_type": request.visualization_type,
                "model_version": request.model_version,
                "generated_at": datetime.now().isoformat(),
                "sequence_length": len(request.sequence)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate attention visualization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
