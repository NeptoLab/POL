#!/usr/bin/env python3

import random
from typing import List, Optional
from datasets import load_dataset
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    path: str
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    text_column: str = "text"
    description: str = ""

class DataAcquisition:
    def __init__(self):
        # No explicit cache_dir; use default HuggingFace cache
        self.available_datasets = {
            "wikipedia_en": DatasetConfig(
                name="Wikipedia English",
                path="wikimedia/wikipedia",
                subset="20231101.en",
                text_column="text",
                description="English Wikipedia articles"
            ),
            "wikipedia_simple": DatasetConfig(
                name="Wikipedia Simple English",
                path="wikimedia/wikipedia",
                subset="20231101.simple",
                text_column="text",
                description="Simple English Wikipedia"
            ),
            # ... other configs as before ...
        }

    def list_available_datasets(self) -> None:
        print("Available datasets:")
        for key, cfg in self.available_datasets.items():
            print(f"  {key}: {cfg.name} – {cfg.description}")

    def load_dataset(self, key: str):
        if key not in self.available_datasets:
            raise ValueError(f"Unknown dataset '{key}'")
        cfg = self.available_datasets[key]
        logger.info(f"Loading dataset {cfg.name}")
        params = {
            "path": cfg.path,
            "split": cfg.split,
            "streaming": cfg.streaming,
        }
        if cfg.subset:
            params["name"] = cfg.subset
        ds = load_dataset(**params)
        return ds

    def get_text_samples(
        self,
        dataset,
        num_samples: int = 1000,
        text_column: Optional[str] = None,
        min_length: int = 100
    ) -> List[str]:
        col = text_column or getattr(dataset, 'column_names', [None])[0] or 'text'
        samples = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item.get(col, "")
            if isinstance(text, str) and len(text) >= min_length:
                samples.append(text)
        logger.info(f"Collected {len(samples)} samples from {dataset}" )
        return samples

    def load_datasets(
        self,
        dataset_keys: List[str],
        samples_per_dataset: Optional[int] = None,
        total_samples: Optional[int] = None
    ) -> List[str]:
        all_samples = []
        for key in dataset_keys:
            try:
                ds = self.load_dataset(key)
                cfg = self.available_datasets[key]
                count = samples_per_dataset or (total_samples or 1000)
                samples = self.get_text_samples(
                    ds,
                    num_samples=count,
                    text_column=cfg.text_column
                )
                all_samples.extend(samples)
                if total_samples and len(all_samples) >= total_samples:
                    break
            except Exception as e:
                logger.error(f"Error loading {key}: {e}")
        if total_samples:
            all_samples = all_samples[:total_samples]
        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def save_to_file(self, texts: List[str], filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t.replace('\n', ' ').strip() + '\n')
        logger.info(f"Saved {len(texts)} texts to {filename}")

    def load_from_file(self, filename: str) -> List[str]:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(lines)} texts from {filename}")
        return lines

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Acquisition Module")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--datasets", nargs='+')
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--per", type=int, help="samples per dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    da = DataAcquisition()
    if args.list:
        da.list_available_datasets()
        exit()
    if not args.datasets:
        print("Specify --datasets")
        exit()
    texts = da.load_datasets(
        args.datasets,
        samples_per_dataset=args.per,
        total_samples=args.samples
    )
    if args.output:
        da.save_to_file(texts, args.output)
    else:
        print(f"Loaded {len(texts)} samples")
