#!/usr/bin/env python3
"""
Standalone inference script for MusicGen Remixer without Cog dependencies.
Uses Meta's pre-trained melody model by default.
"""

import os
import random
import sys
from typing import Optional, List
import argparse
from pathlib import Path

import torchaudio
import typing as tp
import numpy as np
import torch

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

import librosa
import subprocess
import math

import allin1
import pytsmod as tsm


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MusicGenRemixer:
    def __init__(self, device=None):
        """Initialize the MusicGen Remixer with Meta's melody model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mbd = None
        print(f"Using device: {self.device}")
    
    def setup(self):
        """Load the model into memory."""
        print("Loading MusicGen melody model...")
        # Load Meta's pre-trained melody model instead of custom chord model
        self.model = MusicGen.get_pretrained('facebook/musicgen-melody', device=self.device)
        self.mbd = MultiBandDiffusion.get_mbd_musicgen()
        print("Model loaded successfully!")
    
    def separate_vocals(self, music_input, sr):
        """Separate vocals from music using demucs."""
        from demucs.audio import convert_audio
        from demucs.apply import apply_model
        
        # Use a standard demucs model for vocal separation
        import demucs.pretrained
        demucs_model = demucs.pretrained.get_model('htdemucs')
        demucs_model = demucs_model.to(self.device)
        
        wav = convert_audio(music_input, sr, demucs_model.samplerate, demucs_model.audio_channels)
        stems = apply_model(demucs_model, wav, device=self.device)
        
        # Combine non-vocal stems as background
        background = (stems[:, demucs_model.sources.index('drums')] + 
                     stems[:, demucs_model.sources.index('other')] + 
                     stems[:, demucs_model.sources.index('bass')])
        vocals = stems[:, demucs_model.sources.index('vocals')]
        
        # Convert to model sample rate
        background = convert_audio(background, demucs_model.samplerate, self.model.sample_rate, 1)
        vocals = convert_audio(vocals, demucs_model.samplerate, self.model.sample_rate, 1)
        
        return vocals, background

    def remix_music(
        self,
        prompt: str,
        music_input_path: str,
        output_path: str = "output.wav",
        multi_band_diffusion: bool = False,
        normalization_strategy: str = "loudness",
        beat_sync_threshold: Optional[float] = None,
        chroma_coefficient: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        classifier_free_guidance: int = 3,
        return_instrumental: bool = False,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate a music remix based on the input audio and prompt.
        
        Args:
            prompt: Description of the music to generate
            music_input_path: Path to input audio file
            output_path: Path for output audio file
            multi_band_diffusion: Whether to use MultiBand Diffusion for decoding
            normalization_strategy: Strategy for normalizing audio
            beat_sync_threshold: Threshold for beat synchronization
            chroma_coefficient: Coefficient for chord chroma conditioning
            top_k: Reduces sampling to k most likely tokens
            top_p: Reduces sampling to tokens with cumulative probability p
            temperature: Controls sampling diversity
            classifier_free_guidance: Influence of inputs on output
            return_instrumental: Whether to return instrumental version
            seed: Random seed for reproducibility
            
        Returns:
            List of paths to generated audio files
        """
        
        if not prompt:
            raise ValueError("Must provide `prompt`.")
        if not os.path.exists(music_input_path):
            raise ValueError(f"Input file {music_input_path} does not exist.")
        
        # Clean up any existing output directories
        for dir_name in ['demix', 'spec']:
            if os.path.isdir(dir_name):
                import shutil
                shutil.rmtree(dir_name)
        
        # Set random seed
        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
        set_all_seeds(seed)
        print(f"Using seed {seed}")
        
        # Music Structure Analysis
        print("Analyzing input music structure...")
        music_input_analysis = allin1.analyze(music_input_path)
        
        # Load audio
        music_input, sr = torchaudio.load(music_input_path)
        print(f"BPM: {music_input_analysis.bpm}")
        
        # Set beat sync threshold
        if not beat_sync_threshold or beat_sync_threshold == -1:
            if music_input_analysis.bpm is not None:
                beat_sync_threshold = 1.1 / (int(music_input_analysis.bpm) / 60)
            else:
                beat_sync_threshold = 0.75
        
        # Add BPM to prompt if available
        if music_input_analysis.bpm is not None:
            prompt = prompt + f', bpm : {int(music_input_analysis.bpm)}'
        
        # Prepare audio dimensions
        music_input = music_input[None] if music_input.dim() == 2 else music_input
        duration = music_input.shape[-1] / sr
        wav_sr = self.model.sample_rate
        
        # Separate vocals and background
        print("Separating vocals from music...")
        vocal, background = self.separate_vocals(music_input, sr)
        
        # Save vocal track
        audio_write(
            "input_vocal",
            vocal[0].cpu(),
            self.model.sample_rate,
            strategy=normalization_strategy,
        )
        
        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )
        
        # Generate music
        print("Generating music...")
        with torch.no_grad():
            # For Meta's melody model, we use the melody conditioning
            wav = self.model.generate_with_chroma([prompt], music_input, sr, progress=True)
            if multi_band_diffusion:
                # Note: MBD requires tokens, but standard MusicGen API doesn't return them
                print("Warning: MultiBand Diffusion not supported with standard melody model")
        
        # Clean up NaN and inf values
        mask_nan = torch.isnan(wav)
        mask_inf = torch.isinf(wav)
        wav[mask_nan] = 0  
        wav[mask_inf] = 1
        
        # Normalize amplitude
        wav_amp = wav.abs().max()
        if wav_amp > 0:
            wav = (wav / wav_amp).cpu()
        
        # Save background track
        audio_write(
            "background",
            wav[0].cpu(),
            self.model.sample_rate,
            strategy=normalization_strategy,
        )
        
        wav_length = wav.shape[-1]
        
        # Beat synchronization
        if len(music_input_analysis.downbeats) > 0:
            print("Performing beat synchronization...")
            wav_analysis = allin1.analyze('background.wav')
            
            wav_downbeats = []
            input_downbeats = []
            
            music_input_downbeats = music_input_analysis.downbeats
            
            for wav_beat in wav_analysis.downbeats:
                input_beat = min(music_input_downbeats, key=lambda x: abs(wav_beat - x), default=None)
                if input_beat is None:
                    continue
                    
                print(f"Syncing beat: {wav_beat:.2f} -> {input_beat:.2f}")
                
                if len(input_downbeats) != 0 and int(input_beat * wav_sr) == input_downbeats[-1]:
                    print('Dropped duplicate beat')
                    continue
                    
                if abs(wav_beat - input_beat) > beat_sync_threshold:
                    input_beat = wav_beat
                    print('Beat replaced due to threshold')
                    
                wav_downbeats.append(int(wav_beat * wav_sr))
                input_downbeats.append(int(input_beat * wav_sr))
            
            # Apply time stretching for beat synchronization
            if wav_downbeats and input_downbeats:
                downbeat_offset = input_downbeats[0] - wav_downbeats[0]
                
                if downbeat_offset > 0:
                    channel = wav.shape[1]
                    wav = torch.concat([torch.zeros([1, channel, int(downbeat_offset)]).cpu(), wav.cpu()], dim=-1)
                    for i in range(len(wav_downbeats)):
                        wav_downbeats[i] = wav_downbeats[i] + downbeat_offset
                
                wav_downbeats = [0] + wav_downbeats + [wav_length]
                input_downbeats = [0] + input_downbeats + [wav_length]
                
                wav = torch.Tensor(tsm.wsola(
                    wav[0].cpu().detach().numpy(), 
                    np.array([wav_downbeats, input_downbeats])
                ))[..., :wav_length].unsqueeze(0).to(torch.float32)
                
                # Clean up again after time stretching
                mask_nan = torch.isnan(wav)
                mask_inf = torch.isinf(wav)
                wav[mask_nan] = 0
                wav[mask_inf] = 1
                
                wav_amp = wav.abs().max()
                if wav_amp != 0:
                    wav = (wav / wav_amp).cpu()
                
                audio_write(
                    "background_synced",
                    wav[0].cpu(),
                    self.model.sample_rate,
                    strategy=normalization_strategy,
                    loudness_compressor=True,
                )
        
        # Mix with vocals
        wav = wav.to(torch.float32)
        wav_amp = wav.abs().max()
        vocal_amp = vocal.abs().max()
        
        if wav_amp > 0 and vocal_amp > 0:
            wav = 0.5 * (wav / wav_amp).cpu() + 0.5 * (vocal / vocal_amp)[..., :wav_length].cpu() * 0.5
        
        # Final cleanup and normalization
        mask_nan = torch.isnan(wav)
        mask_inf = torch.isinf(wav)
        wav[mask_nan] = 0  
        wav[mask_inf] = 1
        
        wav_amp = wav.abs().max()
        if wav_amp != 0:
            wav = (wav / wav_amp).cpu()
        
        # Write final output
        audio_write(
            Path(output_path).stem,
            wav[0].cpu(),
            self.model.sample_rate,
            strategy=normalization_strategy,
            loudness_compressor=True,
        )
        
        output_files = [output_path]
        
        if return_instrumental:
            inst_path = "background_synced.wav" if os.path.exists("background_synced.wav") else "background.wav"
            if os.path.exists(inst_path):
                output_files.append(inst_path)
        
        print(f"Generated audio saved to: {output_files}")
        return output_files


def main():
    parser = argparse.ArgumentParser(description='MusicGen Remixer - Standalone Inference')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Description of the music to generate')
    parser.add_argument('--music_input', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--multi_band_diffusion', action='store_true',
                        help='Use MultiBand Diffusion for decoding (if supported)')
    parser.add_argument('--normalization_strategy', type=str, default='loudness',
                        choices=['loudness', 'clip', 'peak', 'rms'],
                        help='Strategy for normalizing audio')
    parser.add_argument('--beat_sync_threshold', type=float, default=None,
                        help='Threshold for beat synchronization')
    parser.add_argument('--chroma_coefficient', type=float, default=1.0,
                        help='Coefficient for chord chroma conditioning')
    parser.add_argument('--top_k', type=int, default=250,
                        help='Reduces sampling to k most likely tokens')
    parser.add_argument('--top_p', type=float, default=0.0,
                        help='Reduces sampling to tokens with cumulative probability p')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Controls sampling diversity')
    parser.add_argument('--classifier_free_guidance', type=int, default=3,
                        help='Influence of inputs on output')
    parser.add_argument('--return_instrumental', action='store_true',
                        help='Also return instrumental version')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        # Initialize remixer
        remixer = MusicGenRemixer(device=args.device)
        remixer.setup()
        
        # Generate remix
        output_files = remixer.remix_music(
            prompt=args.prompt,
            music_input_path=args.music_input,
            output_path=args.output,
            multi_band_diffusion=args.multi_band_diffusion,
            normalization_strategy=args.normalization_strategy,
            beat_sync_threshold=args.beat_sync_threshold,
            chroma_coefficient=args.chroma_coefficient,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            classifier_free_guidance=args.classifier_free_guidance,
            return_instrumental=args.return_instrumental,
            seed=args.seed,
        )
        
        print("Success! Generated files:")
        for file in output_files:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()