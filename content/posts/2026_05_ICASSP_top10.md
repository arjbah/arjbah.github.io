---
author: "Arjun Bahuguna"
title: "Top 5 Takeaways from ICASSP 2026"
date: "2026-05-09"
description: ""
tags: ["conference takeaways", "music information retrieval", "large audio language models"]
ShowToc: true
TocOpen: false
---

I attended the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) in Barcelona 🟨🟥🟨🟥. These are my five main takeaways, drawn from the papers, tutorials and workshops I attended.

## 1. Audio LLMs are being probed, not only benchmarked

I counted 14+ papers that looked inside large audio language models instead of adding another leaderboard.

- Morais & Fuentes applied Shapley-value decomposition (MM-SHAP) to measure how much each modality contributes in music tasks. Audio and text contribute unevenly across subtasks.
- Chen used variational inference to calibrate uncertainty in audio question answering.
- Tam & Chen ran a controlled study that found social bias in audio LLM clinical decision-making when the speaker's voice characteristics varied.
- Kim et al. released an auditory illusion benchmark that exposed systematic failure modes in large audio language models.
- Sridhar et al. (Audiocards) showed that structured metadata injected at inference time significantly improves text-audio retrieval.

Taken together, these suggest the models lean heavily on linguistic scaffolding rather than on audio features alone.

## 2. Flow matching is replacing diffusion as the generative backbone

Flow matching showed up in 25 papers. Continuous normalizing flows with optimal transport trajectories are now used wherever diffusion was two years ago.

- Welker et al. demonstrated real-time streaming mel vocoding.
- Hsieh & Braun applied it to speech restoration under hard latency budgets.
- Vosoughi et al. used latent rectified flow matching for multimodal room impulse response synthesis.
- Pankov et al. (PFluxTTS) combined flow matching with inference-time model fusion for cross-lingual voice cloning.
- Chen et al. (F5E-TTS) aligned semantic representations in a diffusion-transformer backbone.
- Zhang et al. (AnyAccomp) applied it to accompaniment generation through a quantized melodic bottleneck.

Most of these are latency-sensitive tasks, so I think inference efficiency is the main reason for the switch, more than expressiveness.

For the diffusion side, Stanley Chan's tutorial on Diffusion Models for Imaging and Vision traced the path from VAEs to DDPMs to score-based methods (score matching and Langevin dynamics), and then unified them through SDEs. If I could get the recording of one session from Day 1, it would be this one. He annotated the slides with a stylus as he went and made dense math easy to follow.

## 3. Neural audio codecs are becoming shared infrastructure

Codec design (11 papers) has moved from pure compression toward learned representations that serve several purposes.

- Zhang et al. (STACodec) separated semantic and acoustic tokens via distillation, so the fidelity-semantics tradeoff can be controlled at decode time.
- Wang et al. (SwitchCodec) used residual-expert sparse quantization for variable-bitrate coding.
- Aihara et al. (Sunac) built source separation and speech enhancement directly into the codec.
- Chen et al. (AUV) proposed a single nested codebook for a unified audio representation across domains.

Several TTS and generation pipelines now use codec tokens as their main intermediate representation, so the codec architecture is a design choice that affects everything downstream.

The Low-resource Audio Codec Workshop was built around this. The challenge asked for neural audio codecs in two settings: a low bitrate budget of up to 6 kbps and an ultra-low budget of up to 1 kbps. I may be biased, since I supervised a paper accepted to the workshop ("Attention-guided Audio Compression for Multimodal LLMs" by Rane et al.), but the poster session had some excellent ideas. Ziqian Wu from ByteDance won Track 1 of the LRAC Challenge with a system that used SnakeBeta activations, Conv2FormerBlocks and 8 loss terms, reaching a MUSHRA score of 82/100 on clean speech in the low bitrate setting. More here: [LRAC Challenge](https://lnkd.in/esHdD6Sx).

## 4. Music analysis runs on learned, multi-modal representations

The tutorial on Current Approaches to Computational Analysis of Music Audio Signals, by Xavier Serra, Martín Rocamora, Dmitry Bogdanov, Andrea Poltronieri and Pablo Alonso Jiménez, focused on representation learning for music analysis. It traced the move from handcrafted features to deep learning and self-supervised approaches that give large-scale, transferable audio representations. It also covered multi-modal methods that combine audio, symbolic and textual data for music understanding and cross-modal retrieval. [Code](https://lnkd.in/eXbWVJ7v).

## 5. Causal representation learning is worth watching

Ali Tajer's tutorial on Causal Representation Learning (CRL) covered a framework for recovering interpretable, mechanistic representations from high-dimensional data. It went through disentanglement, causal inference and identifiability, and methods such as interventional CRL, which uses controlled perturbations to recover causal structure. Prof. Tajer compared unsupervised CRL with self-supervised JEPA on a robotics task, and pointed out that CRL could be applied to blind source separation to help with generalization and interpretability. [Slides](https://lnkd.in/eSPPvUdx).

## Beyond the talks

The welcome reception had seafood, castells and a great live band, and it's hard to pick a favorite. ICASSP started with a bang.

If you want to explore more trends, Hieu-Thi Luong's [paper theme map](https://lnkd.in/eD9tx_Ze) is a good place to start. I'm happy to go deeper on any of these, and I'd like to hear your own takeaways from ICASSP 2026.
