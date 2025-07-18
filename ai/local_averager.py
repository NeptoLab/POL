#!/usr/bin/env python3

import os
import sys
import asyncio
import hashlib
import pickle
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import hivemind
from hivemind import DHT, get_logger

logger = get_logger(__name__)

DHT_PREFIX = os.getenv("DHT_PREFIX", "model_deltas")
DELTA_KEY_PREFIX = "raw:"
ACK_TTL = 3600
LOSS_GATE_THRESHOLD = 0.999
MIN_LOSS_SKIP = 0.05
AGG_INTERVAL = int(os.getenv("AGG_INTERVAL", 60))

DEV_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
    "Distributed training enables scaling across multiple devices to handle larger models and datasets efficiently.",
    "Federated learning preserves data privacy while enabling collaborative model training across distributed clients.",
    "Neural networks learn complex patterns from data through iterative backpropagation and weight optimization.",
    "Deep learning models require substantial computational resources for training on large-scale datasets.",
    "Natural language processing enables computers to understand, interpret, and generate human language effectively.",
    "Computer vision algorithms can identify objects, faces, and patterns in digital images and video streams.",
    "Reinforcement learning agents learn optimal behaviors through trial-and-error interactions with environments.",
    "Transfer learning allows models to leverage knowledge from pre-trained networks for new tasks.",
    "Gradient descent optimization algorithms minimize loss functions to improve model performance iteratively.",
    "Convolutional neural networks excel at processing grid-like data such as images and feature maps.",
    "Recurrent neural networks can process sequential data like text, speech, and time series effectively.",
    "Attention mechanisms help models focus on relevant parts of input sequences for better predictions.",
    "Transformer architectures revolutionized natural language processing with self-attention and parallel computation.",
    "Generative adversarial networks consist of competing generator and discriminator networks for data synthesis.",
    "Autoencoders learn compressed representations of data for dimensionality reduction and feature extraction.",
    "Batch normalization stabilizes training by normalizing inputs to each layer during network optimization.",
    "Dropout regularization prevents overfitting by randomly setting neuron outputs to zero during training.",
    "Cross-validation techniques help evaluate model performance and generalization on unseen data samples.",
    "Hyperparameter tuning optimizes model architecture and training parameters for better performance.",
    "Data augmentation increases training dataset size through transformations like rotation, scaling, and cropping.",
    "Feature engineering involves selecting and transforming relevant variables for machine learning algorithms.",
    "Ensemble methods combine multiple models to improve prediction accuracy and reduce overfitting risks.",
    "Principal component analysis reduces data dimensionality while preserving most important variance components.",
    "Clustering algorithms group similar data points together without requiring labeled training examples.",
    "Classification models predict discrete categories or classes based on input feature patterns.",
    "Regression analysis estimates continuous numerical values from input variables using statistical methods.",
    "Decision trees create interpretable models by recursively splitting data based on feature thresholds.",
    "Random forests combine multiple decision trees to improve accuracy and reduce overfitting.",
    "Support vector machines find optimal decision boundaries by maximizing margins between different classes.",
    "K-means clustering partitions data into k clusters by minimizing within-cluster sum of squares.",
    "Linear regression models the relationship between variables using a linear equation approach.",
    "Logistic regression predicts binary outcomes using the sigmoid function for probability estimation.",
    "Naive Bayes classifiers assume feature independence and use probabilistic reasoning for predictions.",
    "Neural network architectures define the structure and connections between layers of artificial neurons.",
    "Activation functions introduce non-linearity into neural networks enabling complex pattern recognition.",
    "Loss functions measure the difference between predicted and actual values during model training.",
    "Optimizers adjust model parameters to minimize loss functions through various gradient-based approaches.",
    "Learning rates control the step size of parameter updates during neural network training processes.",
    "Regularization techniques prevent overfitting by adding penalty terms to the loss function.",
    "Validation sets help monitor model performance and prevent overfitting during training iterations.",
    "Test sets provide unbiased evaluation of final model performance on completely unseen data.",
    "Training sets contain labeled examples used to teach models patterns and relationships in data.",
    "Overfitting occurs when models memorize training data but fail to generalize to new examples.",
    "Underfitting happens when models are too simple to capture underlying patterns in the data.",
    "Bias-variance tradeoff balances model complexity to minimize both systematic errors and prediction variance.",
    "Feature selection identifies the most relevant variables for improving model performance and interpretability.",
    "Dimensionality reduction techniques compress high-dimensional data while preserving important information.",
    "Outlier detection identifies unusual data points that deviate significantly from normal patterns."
] * 11

class LocalAverager:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "neptolab/cloady-235B")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing LocalAverager with model: {self.model_name}")
        self._load_model()
        self._setup_dht()

    def _load_model(self):
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.base_model.eval()

    def _setup_dht(self):
        genesis = os.getenv("GENESIS_HOST")
        peers = []
        if genesis and genesis != "localhost":
            peers = [f"/ip4/{genesis}/tcp/13337"]
        self.dht = DHT(
            start=True,
            initial_peers=peers,
            client_mode=False,
            prefix=DHT_PREFIX
        )
        logger.info("DHT initialized for averager")

    async def compute_loss(self, model: torch.nn.Module, texts: List[str]) -> float:
        def _inner():
            total, count = 0.0, 0
            for text in texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=256, padding=True
                ).to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                total += outputs.loss.item()
                count += 1
            return total / count if count else float('inf')
        
        return await asyncio.get_event_loop().run_in_executor(None, _inner)

    def apply_lora_delta(self, delta_weights: Dict[str, torch.Tensor]) -> torch.nn.Module:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"]
        )
        model = get_peft_model(self.base_model, lora_config)
        for name, param in model.named_parameters():
            if name in delta_weights:
                param.data = delta_weights[name].to(param.device, dtype=param.dtype)
        model.eval()
        return model

    async def loss_gate_validation(self, delta_weights: Dict[str, torch.Tensor]) -> bool:
        loss_before = await self.compute_loss(self.base_model, DEV_TEXTS[:50])
        if loss_before < MIN_LOSS_SKIP:
            logger.info(f"Skipping gate: loss_before={loss_before:.4f} < {MIN_LOSS_SKIP}")
            return True
        model = self.apply_lora_delta(delta_weights)
        loss_after = await self.compute_loss(model, DEV_TEXTS[:50])
        is_valid = loss_after < LOSS_GATE_THRESHOLD * loss_before
        logger.info(f"Loss gate: before={loss_before:.4f}, after={loss_after:.4f}, valid={is_valid}")
        return is_valid

    async def publish_ack(self, agg_id: str, validator_pubkey: bytes) -> bool:
        try:
            peer_id = hashlib.sha256(validator_pubkey).digest()[:8]
            key = f"ack:{agg_id[:16]}:{peer_id.hex()}"
            success = await self.dht.store(key=key, value=b"\x01", expiration_time=hivemind.get_dht_time()+ACK_TTL)
            logger.info(f"ACK publish {'succeeded' if success else 'failed'}: {key}")
            return success
        except Exception as e:
            logger.error(f"ACK error: {e}")
            return False

    async def aggregate_and_validate(self, raw_ids: List[str], peer_ids: List[str]) -> Optional[str]:
        deltas = []
        for raw_id in raw_ids:
            try:
                entry = await self.dht.get(f"{DELTA_KEY_PREFIX}{raw_id}")
                if entry and entry.value:
                    delta = torch.load(entry.value, map_location='cpu', weights_only=True)
                    deltas.append(delta)
            except Exception as e:
                logger.warning(f"Failed to load delta {raw_id}: {e}")
                continue
        
        if not deltas:
            logger.warning("No deltas fetched for aggregation")
            return None
        
        distances = [(i, sum((delta_i[k]-delta_j[k]).norm().item()**2
                    for j, delta_j in enumerate(deltas) if i!=j for k in delta_i))
                     for i, delta_i in enumerate(deltas)]
        sel = sorted(distances, key=lambda x: x[1])[:min(len(deltas), 3)]
        selected = [deltas[i] for i,_ in sel]
        
        avg = {k: torch.stack([d[k] for d in selected]).mean(dim=0) for k in selected[0]}
        
        if not await self.loss_gate_validation(avg):
            logger.warning("Aggregation failed loss gate")
            return None
        
        agg_id = hashlib.sha256(pickle.dumps(avg)).hexdigest()
        pubkey_hex = os.getenv("VALIDATOR_PUBKEY")
        if pubkey_hex:
            await self.publish_ack(agg_id, bytes.fromhex(pubkey_hex))
        logger.info(f"Aggregation complete: {agg_id[:16]}")
        return agg_id

    async def service_loop(self):
        seen = set()
        async for entry in self.dht.wait_on_prefix(DELTA_KEY_PREFIX):
            key = entry.key
            raw_id = key.replace(DELTA_KEY_PREFIX, "")
            if raw_id in seen:
                continue
            seen.add(raw_id)
            peer_id = entry.publisher.hex()
            logger.info(f"New delta {raw_id[:16]} from {peer_id}")
            await self.aggregate_and_validate([raw_id], [peer_id])


def main():
    averager = LocalAverager()
    try:
        asyncio.run(averager.service_loop())
    except KeyboardInterrupt:
        logger.info("Averager stopped by user")

if __name__ == "__main__":
    main()
