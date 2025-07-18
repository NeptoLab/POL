#!/usr/bin/env python3

import os
import io
import argparse
import asyncio
import hashlib

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import hivemind
from hivemind import DHT, get_logger

from data_acquisition import DataAcquisition

logger = get_logger(__name__)

class LoRATrainer:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "neptolab/cloady-235B")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_acquisition = DataAcquisition()

        logger.info(f"Initializing trainer with model: {self.model_name}")
        self._load_model()
        self._setup_lora()
        self._setup_dht()

    def _load_model(self):
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.base_model.to(self.device)

    def _setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    def _setup_dht(self):
        peers = []
        genesis = os.getenv("GENESIS_HOST")
        if genesis and genesis != "localhost":
            peers = [f"/ip4/{genesis}/tcp/13337"]
        self.dht = DHT(
            start=True,
            initial_peers=peers,
            client_mode=False,
            prefix="model_deltas",
            host_maddrs=["/ip4/0.0.0.0/tcp/0"]
        )
        logger.info(f"DHT initialized. Peers: {peers}, Peer ID: {self.dht.peer_id}")

    def extract_delta(self) -> bytes:
        delta = {
            name: param.data.cpu().half()
            for name, param in self.model.named_parameters()
            if name.startswith("lora_") and param.requires_grad
        }
        buffer = io.BytesIO()
        torch.save(delta, buffer)
        buffer.seek(0)
        data = buffer.getvalue()
        if len(data) > 150 * 1024:
            logger.warning(f"Delta size {len(data)} bytes exceeds 150KB")
        return data

    async def publish(self, buf: bytes) -> str:
        key = hashlib.sha256(buf).hexdigest()
        success = await self.dht.store(
            key=f"raw:{key}",
            value=buf,
            expiration_time=hivemind.get_dht_time() + 3600
        )
        if success:
            logger.info(f"Published delta {key[:16]} to DHT")
        else:
            logger.warning(f"Failed to publish delta {key[:16]}")
        return key

    def train_step(self, texts: list) -> float:
        total_loss = 0.0
        for text in texts:
            batch = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(texts) if texts else 0.0

    async def loop(self, steps: int = 100, num_samples: int = None):
        keys = list(self.data_acquisition.available_datasets.keys())
        data = self.data_acquisition.load_datasets(
            dataset_keys=keys,
            total_samples=num_samples
        ) if num_samples else self.data_acquisition.load_datasets(dataset_keys=keys)

        if not data:
            logger.error("No training data available")
            return

        logger.info(f"Starting training: {steps} steps on {len(data)} samples")
        batch_size = 4
        for step in range(steps):
            start = (step * batch_size) % len(data)
            batch = data[start : start + batch_size]
            loss = self.train_step(batch)
            if step % 10 == 0:
                delta_buf = self.extract_delta()
                key = await self.publish(delta_buf)
                logger.info(f"Step {step}, loss={loss:.4f}, delta={key[:16]}")
        logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Trainer with Hivemind DHT")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--num-samples", type=int,
        help="Limit total samples loaded from datasets"
    )
    args = parser.parse_args()

    trainer = LoRATrainer()
    asyncio.run(trainer.loop(steps=args.steps, num_samples=args.num_samples))
