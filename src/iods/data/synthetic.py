"""Synthetic species dataset for smoke testing."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from iods.encoders.dna_encoder import kmer_frequency_spectrum


N_FAMILIES = 4
N_GENERA_PER_FAMILY = 3
N_SPECIES_PER_GENUS = 5


@dataclass
class SyntheticSpecies:
    species_id: str
    family: str
    genus: str
    dna: str
    image: np.ndarray
    audio: np.ndarray
    timeseries: np.ndarray
    phenotype: np.ndarray
    metadata: np.ndarray
    environment: np.ndarray


def _random_dna_sequence(length, family_seed, rng):
    bases = ["A", "C", "G", "T"]
    gc_target = 0.4 + 0.05 * (family_seed % 5)
    weights = [(1 - gc_target) / 2, gc_target / 2,
               gc_target / 2, (1 - gc_target) / 2]
    return "".join(rng.choices(bases, weights=weights, k=length))


def make_synthetic_dataset(
    seed: int = 0,
    n_families: int = N_FAMILIES,
    n_genera_per_family: int = N_GENERA_PER_FAMILY,
    n_species_per_genus: int = N_SPECIES_PER_GENUS,
    dna_length: int = 200,
    image_size: int = 16,
    audio_mels: int = 16,
    audio_frames: int = 16,
    ts_length: int = 64,
    phenotype_dim: int = 16,
    meta_dim: int = 8,
    env_dim: int = 8,
):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    species_list = []
    for f_idx in range(n_families):
        family = f"family_{f_idx:02d}"
        family_latent = np_rng.standard_normal(8)
        for g_idx in range(n_genera_per_family):
            genus = f"genus_{f_idx:02d}_{g_idx:02d}"
            genus_latent = family_latent + 0.3 * np_rng.standard_normal(8)
            for s_idx in range(n_species_per_genus):
                species_id = f"sp_{f_idx:02d}_{g_idx:02d}_{s_idx:02d}"
                species_latent = genus_latent + 0.15 * np_rng.standard_normal(8)
                dna = _random_dna_sequence(dna_length, f_idx, rng)
                base_color = np.tanh(species_latent[:3])
                image = (
                    np.tile(base_color.reshape(3, 1, 1),
                            (1, image_size, image_size))
                    + 0.1 * np_rng.standard_normal((3, image_size, image_size))
                ).astype(np.float32)
                audio = (
                    species_latent[3] * 0.3
                    + 0.2 * np_rng.standard_normal((1, audio_mels, audio_frames))
                ).astype(np.float32)
                t = np.linspace(0, 1, ts_length)
                freq = 2.0 + 3.0 * np.tanh(species_latent[4])
                ts = (
                    np.sin(2 * np.pi * freq * t).reshape(1, -1)
                    + 0.1 * np_rng.standard_normal((1, ts_length))
                ).astype(np.float32)
                W = np_rng.standard_normal((phenotype_dim, 8))
                phenotype = (W @ species_latent
                             + 0.05 * np_rng.standard_normal(phenotype_dim)).astype(np.float32)
                metadata = np.zeros(meta_dim, dtype=np.float32)
                metadata[: min(meta_dim, 4)] = species_latent[: min(meta_dim, 4)]
                environment = np_rng.standard_normal(env_dim).astype(np.float32)
                species_list.append(SyntheticSpecies(
                    species_id=species_id, family=family, genus=genus, dna=dna,
                    image=image, audio=audio, timeseries=ts,
                    phenotype=phenotype, metadata=metadata, environment=environment,
                ))
    return species_list


class SyntheticIODSDataset(Dataset):
    def __init__(self, species, modality_dropout_prob: float = 0.0, seed: int = 0):
        self.species = species
        self.modality_dropout_prob = modality_dropout_prob
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.species)

    def __getitem__(self, idx):
        sp = self.species[idx]
        sample = {
            "species_id": sp.species_id,
            "family": sp.family,
            "genus": sp.genus,
            "dna_string": sp.dna,
            "image": torch.from_numpy(sp.image),
            "audio": torch.from_numpy(sp.audio),
            "timeseries": torch.from_numpy(sp.timeseries),
            "phenotype": torch.from_numpy(sp.phenotype),
            "metadata": torch.from_numpy(sp.metadata),
            "environment": torch.from_numpy(sp.environment),
            "kmer_target": kmer_frequency_spectrum(sp.dna, k=4),
        }
        if self.modality_dropout_prob > 0:
            for key in ("image", "audio", "timeseries", "phenotype",
                        "metadata", "environment"):
                if self._rng.random() < self.modality_dropout_prob:
                    sample[key] = None
        return sample


def collate_synthetic_batch(samples):
    out = {}
    out["species_id"] = [s["species_id"] for s in samples]
    out["family"] = [s["family"] for s in samples]
    out["genus"] = [s["genus"] for s in samples]
    out["dna_strings"] = [s["dna_string"] for s in samples]
    out["kmer_target"] = torch.stack([s["kmer_target"] for s in samples])
    for key in ("image", "audio", "timeseries", "phenotype", "metadata", "environment"):
        if all(s.get(key) is not None for s in samples):
            out[key] = torch.stack([s[key] for s in samples])
        else:
            out[key] = None
    return out
