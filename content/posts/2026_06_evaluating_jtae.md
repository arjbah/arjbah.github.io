---
author: "Arjun Bahuguna"
title: "Embedding Geometry in Pretrained CLAP Models"
date: "2026-05-27"
description: "A long-form walkthrough of why pretrained CLAP models fail on compositional audio–text benchmarks, framed through the geometry of their embedding spaces: RankMe effective rank, Two-NN intrinsic dimension, residual singular-value energy, and the audio–text modality gap."
tags: ["evaluation", "self-supervised learning", "music-ir", "audio-language-models", "CLAP", "embedding-geometry"]
ShowToc: true
TocOpen: true
Draft: false
---

Contrastive Language–Audio Pretraining (CLAP) models have, in the space of about three years, become the default backbone for zero-shot audio classification, audio–text retrieval, and a long tail of downstream Music Information Retrieval (MIR) tasks ([Wu et al., 2023](#references); [Li et al., 2024](#references); [Niizumi et al., 2024](#references)). They are appealing for the same reason CLIP was appealing in vision: a single pair of encoders, trained once on web-scale audio–caption pairs, exposes a joint embedding space in which arbitrary natural-language queries can be compared to arbitrary audio clips with a dot product. No task-specific heads, no per-task fine-tuning, no labelled training data downstream — just two encoders and a similarity matrix.

The catch is that this story works beautifully for *coarse* queries — "a dog barking", "acoustic guitar", "applause in a concert hall" — and noticeably less well for queries that require the model to actually *compose* its vocabulary. The CompA benchmark ([Ghosh et al., 2024](#references)) makes this concrete with two deceptively simple tasks. In **CompA-Order**, a pair of clips and a pair of captions share exactly the same content words but swap the order of two acoustic events ("a dog barks and then a car horn honks" vs. "a car horn honks and then a dog barks"). In **CompA-Attribute**, two clips and two captions share the same set of objects but swap an attribute binding ("a loud dog and a quiet car" vs. "a quiet dog and a loud car"). A model that has actually learned to compose should be near-perfect on both; a model that operates as a bag-of-words should be at chance, because chance is exactly what a bag-of-words gets when the bags are identical.

State-of-the-art CLAP models, as reported in the original CompA paper and as I confirm independently below, are much closer to bag-of-words than to compositional ([Ghosh et al., 2024](#references)). The interesting question is *why*. There are at least three candidate stories. Maybe the audio encoder discards the temporal or attribute information that would let the model tell the two clips apart. Maybe the text encoder collapses the two captions into the same point in embedding space. Maybe the joint contrastive head fails to align audio and text along the axis that actually separates the compositions, even when both encoders carry the relevant signal in isolation.

This post walks through a geometric analysis of three publicly released CLAP variants — LAION-CLAP ([Wu et al., 2023](#references)), MGA-CLAP ([Li et al., 2024](#references)), and M2D-CLAP ([Niizumi et al., 2024](#references)) — on CompA-Order and CompA-Attribute, and asks: can the *shape* of the embedding space, measured without any labels, predict how well the model performs on a hard compositional benchmark? The answer, across the six (model × benchmark) cells I evaluate, is yes — and surprisingly, the two probes that matter are both on the *text* side and the *joint* side, not on the audio side. Text-side effective rank ([Garrido et al., 2023](#references)) and the audio–text modality angle ([Liang et al., 2022](#references)) jointly account for almost all the between-model variance in the hard compositional score, while audio-side intrinsic dimension ([Facco et al., 2017](#references)) and residual singular-value energy are not statistically discriminative on this benchmark.

This is, I think, a useful piece of news for CLAP design. It says that compositional failures in current CLAP models are not primarily an audio-encoder collapse problem; they are a text-space anisotropy and cross-modal alignment problem. And it says that you can detect them on unlabelled embeddings, cheaply, before you ever run a downstream benchmark.

The rest of this post unpacks that claim in depth. I will spend a fair amount of space on the background that makes the geometric probes interpretable — what RankMe is actually measuring, why Two-NN is the right intrinsic-dimension estimator for embedding clouds, what the modality gap is and is not — because the punchline relies on reading these quantities correctly.

## Background: Contrastive Language–Audio Pretraining

### The CLAP recipe

Every CLAP variant I consider here follows the same broad recipe, inherited almost directly from CLIP. There is an audio encoder $f_a: \mathcal{A} \to \mathbb{R}^d$ that maps a waveform (or a mel-spectrogram derived from it) to a $d$-dimensional embedding, and a text encoder $f_t: \mathcal{T} \to \mathbb{R}^d$ that maps a caption to a $d$-dimensional embedding in the same space. Both encoders end with a projection head that maps to the joint space, and both outputs are $\ell_2$-normalised so that the relevant similarity is the cosine

$$
\mathrm{sim}(a, t) = \frac{f_a(a)^\top f_t(t)}{\lVert f_a(a) \rVert_2 \, \lVert f_t(t) \rVert_2}.
$$

Training uses a symmetric InfoNCE objective ([Radford et al., 2021](#references)) on a batch of $N$ matched audio–text pairs $\{(a_i, t_i)\}_{i=1}^N$. Letting $s_{ij} = \mathrm{sim}(f_a(a_i), f_t(t_j)) / \tau$ for some learned temperature $\tau$, the loss is

$$
\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{\exp(s_{ii})}{\sum_{j=1}^N \exp(s_{ij})} + \log \frac{\exp(s_{ii})}{\sum_{j=1}^N \exp(s_{ji})} \right].
$$

The first term inside the bracket is the audio-to-text retrieval loss for the $i$-th audio; the second term is the text-to-audio retrieval loss for the $i$-th caption. Both terms push the matched pair $s_{ii}$ up and pull the off-diagonal entries $s_{ij}$ ($j \neq i$) down. The temperature $\tau$ controls how sharply the softmax distinguishes the positive from the negatives; small $\tau$ means harder negatives matter more.

This objective is *symmetric* in a very specific sense: it cares about ranking the positive pair above other items in the batch. It does *not* care about the absolute geometry of the embedding clouds. It does not say "spread your text embeddings out", or "make sure the audio and text centroids coincide", or "preserve a notion of compositional structure". Anything that ranks the positives correctly is a valid solution, including — and this matters for us — pathological solutions in which the encoders collapse most of their input variation into a small number of effective dimensions.

### Three CLAP variants

The three CLAP variants I evaluate were chosen to span the design space of currently released audio–language models without being a moving target. All three are publicly released and used inference-only; no checkpoint was fine-tuned for this study.

**LAION-CLAP** ([Wu et al., 2023](#references)) pairs an HTSAT-tiny audio backbone ([Chen et al., 2022](#references)) with a RoBERTa-base text encoder ([Liu et al., 2019](#references)) and projects both to a 512-dimensional joint space. It is trained on approximately 630k audio–text pairs aggregated from LAION-Audio-630K, AudioCaps, Clotho, and several other sources, with feature-fusion and keyword-to-caption augmentation. It is the closest analogue to "vanilla CLIP for audio" and is by far the most widely deployed CLAP checkpoint in the literature.

**MGA-CLAP** (Multi-grained Alignment CLAP, [Li et al., 2024](#references)) replaces the global pooled audio–text contrast with a *grouped masked-codebook* contrastive objective that aligns frame-level audio tokens with word-level text tokens through a shared discrete codebook. The intuition is that the standard pooled contrast forces all temporal information into a single vector, throwing away exactly the kind of fine-grained structure that compositional captions are sensitive to. The audio backbone is HTSAT augmented with the codebook; the text encoder is BERT-base ([Devlin et al., 2019](#references)); the joint dimension is 1024.

**M2D-CLAP** ([Niizumi et al., 2024](#references)) takes the opposite approach: it starts from a strong self-supervised audio backbone trained with the Masked Modeling Duo objective ([Niizumi et al., 2023](#references)) and adds a CLAP-style text alignment objective on top, so that the contrastive head does not have to learn audio features from scratch. The audio backbone is an M2D ViT-Base; the text encoder is BERT-base; the joint dimension is 768.

The three models are summarised in Table 1.

| Model       | Audio encoder       | Text encoder  | Dim  |
|-------------|---------------------|---------------|------|
| LAION-CLAP  | HTSAT-tiny          | RoBERTa-base  | 512  |
| MGA-CLAP    | HTSAT + codebook    | BERT-base     | 1024 |
| M2D-CLAP    | M2D ViT-B           | BERT-base     | 768  |

**Table 1.** CLAP variants evaluated, all loaded from public checkpoints.

### Why we should expect compositional trouble

The InfoNCE loss, as written above, has no term that explicitly encourages compositional generalisation. As long as the matched audio–caption pair is closer than the in-batch negatives, the loss is satisfied. In practice, web-scale audio–caption batches contain very few near-paraphrase negatives: most negatives differ from the positive in many content words, so a model that learns a robust "set of nouns" representation gets most of the way to a small loss. The CompA benchmark is, by construction, an adversarial test against exactly this shortcut: the negative caption shares every content word with the positive caption and differs only in word *order* or in attribute *binding*. If a model has used the bag-of-words shortcut during training, it has nothing to fall back on at CompA time.

This is the same diagnosis that has been made repeatedly for vision–language models. VL-Checklist ([Zhao et al., 2022](#references)), ARO ([Yuksekgonul et al., 2023](#references)), and SugarCrepe ([Hsieh et al., 2023](#references)) all report that CLIP-style models behave largely as bags-of-words under attribute, relation, and order swaps. CompA is the audio analogue of this finding. What I want to add is a complementary *why*: not just "the model fails the test", but "here is the geometric signature of the failure on unlabelled data".

## The CompA Benchmark, In Detail

CompA ([Ghosh et al., 2024](#references)) provides two complementary pairwise benchmarks. Both share the same template: each item is a *pair* of audio clips $(a_p, a_r)$ and a *pair* of captions $(c_p, c_r)$, with the convention that $c_p$ matches $a_p$ ("paired") and $c_r$ matches $a_r$ ("reversed"). $c_r$ is constructed from $c_p$ by either swapping the order of two events (Order) or swapping an attribute binding between two objects (Attribute). $a_r$ is a real audio clip that genuinely matches $c_r$ — this is what makes the benchmark non-trivial; it is not a paraphrase test, it is a grounded retrieval test.

CompA reports three scores per benchmark:

- $F$ (text-to-audio): the fraction of items where $c_p$ ranks $a_p$ above $a_r$.
- $G$ (audio-to-text): the fraction of items where $a_p$ ranks $c_p$ above $c_r$.
- $H$ (hard compositional): the fraction of items where *both* directions are correct, i.e. $F$ and $G$ jointly succeed on the same item.

Formally, for the $i$-th pair, let $\mathbb{1}_i^{(F)} = \mathbb{1}\{\mathrm{sim}(c_p^{(i)}, a_p^{(i)}) > \mathrm{sim}(c_p^{(i)}, a_r^{(i)})\}$ and $\mathbb{1}_i^{(G)} = \mathbb{1}\{\mathrm{sim}(a_p^{(i)}, c_p^{(i)}) > \mathrm{sim}(a_p^{(i)}, c_r^{(i)})\}$. Then

$$
F = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_i^{(F)}, \quad G = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_i^{(G)}, \quad H = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_i^{(F)} \cdot \mathbb{1}_i^{(G)}.
$$

CompA-Order contains $N = 400$ such pairs; CompA-Attribute contains $N = 197$. When restricted to the two candidates inside a pair, the chance baseline is $0.5$ for $F$ and $G$, and a model that flips a fair coin in each direction independently would get $H = 0.25$. But the operational CompA evaluation uses the full benchmark candidate pool (all clips and all captions, not just the two in the current pair), and against that pool the chance expectation for $H$ drops to roughly $1/40 \approx 0.025$ for the benchmarks I evaluate. That is the bar a non-trivial compositional system has to clear.

The hard score $H$ is the most informative quantity. A model can be high on $F$ but low on $G$ if the text encoder is well-behaved but the audio encoder cannot tell the two clips apart, and vice versa. $H$ collapses to zero unless both encoders carry the relevant signal *and* the joint space exposes it.

## A Geometric View of CLAP Embeddings

The central move of this work is to compute four geometric probes on the *same* cached embeddings that the retrieval scores are computed on, and to ask whether any of those probes correlate with the hard score $H$ across models.

The four probes are: (1) RankMe effective rank, (2) Two-NN intrinsic dimension, (3) residual singular-value energy, and (4) the audio–text modality gap. Each is computed on a model × benchmark cell — that is, on the audio and text embedding matrices that the retrieval scorer saw for one benchmark with one model. None of them uses labels; all of them are computable on any unlabelled embedding cloud.

### RankMe effective rank

RankMe ([Garrido et al., 2023](#references)) was introduced as a label-free proxy for downstream performance of self-supervised representations. Given an embedding matrix $X \in \mathbb{R}^{N \times D}$ with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(N, D)} \geq 0$, define the singular-value distribution

$$
p_i = \frac{\sigma_i}{\sum_{j} \sigma_j},
$$

and the entropy-based effective rank

$$
\mathrm{rk}(X) = \exp\!\left(-\sum_i p_i \log p_i\right).
$$

This is the exponential of the Shannon entropy of the normalised singular-value distribution. If the singular-value spectrum is concentrated on a single component, $p$ is a point mass, the entropy is zero, and $\mathrm{rk}(X) = 1$. If the spectrum is uniform over all $r$ non-zero components, $\mathrm{rk}(X) = r$. In between, $\mathrm{rk}(X)$ tracks how many directions in $\mathbb{R}^D$ the embedding cloud *meaningfully* occupies, in an entropy-weighted sense.

The reason this is more useful than the bare matrix rank is that the bare matrix rank is almost always equal to $\min(N, D)$ in floating-point arithmetic, which tells you nothing. The reason it is more useful than counting eigenvalues above some threshold is that it does not require a threshold; the entropy formulation does the soft counting for you. ([Garrido et al., 2023](#references)) show empirically that, across a wide range of self-supervised vision models, $\mathrm{rk}(X)$ is positively rank-correlated with downstream linear-probe accuracy. In effect, models that learn representations occupying more dimensions tend to be more useful downstream.

I compute $\mathrm{rk}$ separately on the audio side (concatenating $a_p$ and $a_r$ across all pairs in the benchmark) and on the text side (concatenating $c_p$ and $c_r$).

### Two-NN intrinsic dimension

The Two-NN estimator ([Facco et al., 2017](#references)) is a non-parametric estimator of the local intrinsic dimension of a point cloud. The idea is elegant. For each point $x_i$ in the cloud, let $r_1(i) < r_2(i)$ be its distances to its first and second nearest neighbours, and define the ratio

$$
\mu_i = \frac{r_2(i)}{r_1(i)} \in [1, \infty).
$$

If the cloud is locally a uniform sample from a $d$-dimensional manifold, then $\mu_i$ follows a Pareto distribution with shape parameter $d$:

$$
\Pr(\mu \leq m) = 1 - m^{-d}, \quad m \geq 1.
$$

Taking $-\log$ of both sides of $\Pr(\mu > m) = m^{-d}$ gives $-\log(1 - F(m)) = d \log m$, so fitting a line through the origin in $\log\mu$ versus $-\log(1 - F_{\mathrm{emp}}(\mu))$ space gives an estimate of $d$. In practice the upper tail is noisy, so the standard recipe is to discard the top $\sim 10\%$ of $\mu$ values before fitting.

Two-NN is attractive because it is purely local: it does not require a global manifold model, it does not care about the embedding *dimension* $D$, and it is reasonably robust to noise. For embedding clouds it tends to return values much smaller than $D$ — typical CLIP embeddings live on a $10$ to $20$-dimensional manifold inside a $512$-dimensional ambient space, for instance.

I compute Two-NN separately on the audio side and the text side, on the same concatenated matrices as RankMe.

### Residual singular-value energy

The third probe is the simplest. For an embedding matrix $X$ with singular values $\sigma_i$, the fraction of squared-spectrum energy explained by the top $k$ components is

$$
\rho_k(X) = \frac{\sum_{i \leq k} \sigma_i^2}{\sum_{j} \sigma_j^2},
$$

and the residual energy is $1 - \rho_k(X)$. If $1 - \rho_k(X)$ is small for small $k$ — say $k = 10$ — then most of the cloud's variance is captured by a low-dimensional principal subspace, i.e. the cloud is strongly anisotropic. If $1 - \rho_k(X)$ is large, the cloud spreads its variance more evenly across many directions.

Residual energy is closely related to RankMe but emphasises a different aspect. RankMe is a smooth entropy-weighted count of the spectrum. Residual energy is a hard count above a chosen $k$. The two can disagree on the margins, but they typically point in the same direction.

I report $1 - \rho_{10}^{\text{audio}}$ in the results table. It is a single representative cut; I checked $k \in \{1, 5, 10\}$ and the qualitative ranking is the same.

### The modality gap

The fourth probe is, in some ways, the most interpretable. ([Liang et al., 2022](#references)) noticed that in CLIP-style joint embedding spaces, the image and text centroids are not centred on the origin: they live in two distinct cones, and there is a persistent angular separation between the two modalities, even after $\ell_2$ normalisation and even when the contrastive objective is fully converged. They called this the *modality gap*, and showed that its existence is a generic consequence of the optimisation landscape under a symmetric InfoNCE objective with finite temperature.

For us, let $\bar{a} = \frac{1}{N_a} \sum_i f_a(a_i)$ and $\bar{t} = \frac{1}{N_t} \sum_j f_t(t_j)$ be the audio and text centroids on a given benchmark. Three derived quantities are interesting:

- The $\ell_2$ gap: $\lVert \bar{a} - \bar{t} \rVert_2$.
- The centroid cosine: $\cos\theta = \bar{a}^\top \bar{t} / (\lVert \bar{a} \rVert_2 \lVert \bar{t} \rVert_2)$.
- The angle: $\theta = \arccos(\cos\theta)$, reported in degrees.

The $\ell_2$ gap and the angle are not redundant for unit vectors: if both $\bar{a}$ and $\bar{t}$ are short (i.e. the per-modality clouds are spread out so that their centroids are pulled toward the origin), the $\ell_2$ gap can be small even when the angle is large. As we will see, this distinction matters for the correlation analysis.

## Experimental Setup

The experimental protocol is deliberately spartan. All three CLAP variants are loaded from public checkpoints; no fine-tuning, no hyperparameter search, no prompt engineering. Audio is loaded at the model's native sample rate (32 kHz for LAION-CLAP and MGA-CLAP, 16 kHz for M2D-CLAP), monoised, and truncated or padded to a single $10$-second window. All embeddings are $\ell_2$-normalised before scoring.

M2D-CLAP has a configuration choice worth flagging: its audio path supports either `flat_features=True`, which uses the released $768$-dim semantic projection, or `flat_features=False`, which concatenates a stack of intermediate features into a $3840$-dim representation. I report the $768$-dim variant, which is the configuration the model card recommends for CLAP-style retrieval. The stacked variant exists and would be worth a separate study; my central conclusion below — that text-side geometry and modality angle dominate — would not be affected by the audio-side choice, since the audio-side probes are not the discriminating ones.

Each model is loaded inside an isolated Python environment to avoid dependency clashes between the three upstream codebases (notably `timm` version conflicts between MGA-CLAP and M2D-CLAP). Every retrieval run writes a JSON sidecar containing the model name, checkpoint SHA-256, benchmark identifier, the three scores, the runtime, and the current git SHA. Embeddings are persisted as compressed NPZ so that the geometric probes and the correlation analysis can be re-run without re-encoding.

## Results: Compositional Retrieval

Table 2 reports the three CompA scores for all six (model × benchmark) cells.

| Model       | Benchmark  | $F$    | $G$    | $H$    |
|-------------|------------|--------|--------|--------|
| LAION-CLAP  | Order      | 0.190  | 0.058  | 0.028  |
| LAION-CLAP  | Attribute  | 0.421  | 0.030  | 0.030  |
| MGA-CLAP    | Order      | 0.383  | 0.168  | 0.140  |
| MGA-CLAP    | Attribute  | 0.447  | 0.137  | 0.102  |
| M2D-CLAP    | Order      | 0.373  | 0.190  | 0.125  |
| M2D-CLAP    | Attribute  | 0.457  | 0.086  | 0.076  |

**Table 2.** CompA retrieval scores (higher is better). $H$ is the hard compositional success rate. The random baseline against the full candidate pool is approximately $1/40 = 0.025$.

Two patterns are worth pulling out.

First, LAION-CLAP collapses to the random baseline on $H$ for both benchmarks (0.028 on Order, 0.030 on Attribute). This is consistent with the original CompA paper's finding that vanilla CLAP-style models are essentially bag-of-words on compositional queries ([Ghosh et al., 2024](#references)). $F$ is non-trivial (around 0.2 to 0.4), which says that the text-to-audio direction does carry *some* signal — the matched caption picks the matched audio slightly more often than chance — but $G$ is at floor (0.03 to 0.06), which says that the audio-to-text direction is essentially broken. When you ask "given this audio, which of these two captions is a better match?", LAION-CLAP shrugs.

Second, both MGA-CLAP and M2D-CLAP clear the random baseline on $H$ by a comfortable margin (roughly $5\times$ chance on Order and $3$–$4\times$ chance on Attribute). MGA-CLAP wins on both benchmarks; M2D-CLAP is competitive on Order and falls off on Attribute. The shape of the gap — large on $H$, small on $F$ — tells us that the gain over LAION-CLAP comes mostly from fixing the audio-to-text direction. MGA-CLAP and M2D-CLAP score about $3$–$5\times$ higher on $G$ than LAION-CLAP does, and that improvement is what unlocks $H$.

It is tempting at this point to attribute the gap entirely to the audio encoder, on the reasoning that "$G$ is audio-to-text, so it must be the audio side that matters". That reasoning is wrong, and the geometric analysis shows why.

## Results: Embedding Geometry

Table 3 reports the four geometric probes on the same six cells.

| Model | Bench     | $\mathrm{rk}_{\text{audio}}$ | $\mathrm{rk}_{\text{text}}$ | $d^{\text{2NN}}_{\text{audio}}$ | $d^{\text{2NN}}_{\text{text}}$ | $1 - \rho_{10}^{\text{audio}}$ | $\lVert \bar{a} - \bar{t} \rVert_2$ | angle (°) |
|-------|-----------|------|-------|------|------|------|------|------|
| LAION | Order     | 214.4 | 161.8 | 11.3 | 2.4 | 0.27 | 0.454 | 74.8 |
| LAION | Attribute | 169.1 | 133.6 | 10.0 | 1.2 | 0.18 | 0.460 | 84.0 |
| MGA   | Order     | 369.3 | 316.7 | 10.6 | 2.1 | 0.32 | 0.335 | 48.3 |
| MGA   | Attribute | 225.6 | 199.3 | 9.7  | 1.2 | 0.20 | 0.336 | 50.6 |
| M2D   | Order     | 288.0 | 272.8 | 14.0 | 2.2 | 0.36 | 0.830 | 69.4 |
| M2D   | Attribute | 190.3 | 175.2 | 12.0 | 1.0 | 0.20 | 0.805 | 71.5 |

**Table 3.** Geometric probes on the cached embeddings. $\mathrm{rk}$ is the RankMe effective rank, $d^{\text{2NN}}$ is the Two-NN intrinsic dimension, $1 - \rho_{10}^{\text{audio}}$ is the audio residual singular-value energy after the top $10$ components, and the angle is between the audio and text centroids.

A couple of things jump out before we even compute correlations.

The text-side Two-NN intrinsic dimension is *low* and *tightly clustered* across all models and benchmarks: $d^{\text{2NN}}_{\text{text}}$ ranges from $1.0$ to $2.4$. This is not a bug; it is the geometry of the benchmark. On a CompA benchmark with $N$ pairs, the text side contains exactly $2N$ captions, organised in $N$ paired/reversed pairs that differ in a single swap. The local manifold around any caption is essentially "yourself, and your near-swap partner", which is one-dimensional. Two-NN sees this and correctly returns $\sim 1$ to $2$ across the board. The probe is doing the right thing; it is just that the right thing here is not informative for ranking models.

The audio-side intrinsic dimension is also relatively flat across models ($10$ to $14$). The three models produce audio clouds of broadly comparable shape, even though their backbones are very different (HTSAT vs. HTSAT-with-codebook vs. M2D ViT-B).

The two probes that *do* differentiate the models are RankMe (on both sides, but especially on text) and the modality angle. MGA-CLAP has both the highest effective rank ($\mathrm{rk}_{\text{text}} = 316.7$ on Order, $199.3$ on Attribute) and the smallest modality angle ($48.3°$ on Order, $50.6°$ on Attribute). LAION-CLAP has the lowest effective rank and the largest angle ($74.8°$ to $84.0°$). M2D-CLAP sits in between on rank and is comparable to LAION on angle.

It is also worth noting that the $\ell_2$ modality gap does *not* track the angle. M2D-CLAP has by far the largest $\ell_2$ gap ($0.80$+) but a mid-pack angle ($\sim 70°$). MGA-CLAP has the smallest $\ell_2$ gap and the smallest angle. LAION-CLAP has a small $\ell_2$ gap but the largest angle. This is exactly the distinction I flagged in the modality-gap section: short centroids can produce small $\ell_2$ gaps even when the centroids point in very different directions, and the operative quantity for cross-modal retrieval is the *angle*, not the $\ell_2$ distance.

## Correlations With the Hard Score

Table 4 reports Spearman ($\rho_S$) and Pearson ($r_P$) correlations of each probe with $H$ across the six (model × benchmark) cells.

| Probe                          | $\rho_S$ | $p_S$ | $r_P$ | $p_P$ |
|--------------------------------|----------|-------|-------|-------|
| **RankMe (text)**              | **+0.94** | **0.005** | **+0.92** | **0.008** |
| **Modality cosine**            | **+0.89** | **0.019** | +0.79 | 0.059 |
| **Modality angle**             | **−0.89** | **0.019** | −0.79 | 0.064 |
| RankMe (audio)                 | +0.83 | 0.042 | +0.83 | 0.040 |
| Residual SVD ($k{=}10$)        | +0.31 | 0.544 | −0.05 | 0.928 |
| Modality $\ell_2$ gap          | −0.20 | 0.704 | +0.07 | 0.891 |
| Two-NN ID (audio)              | +0.09 | 0.872 | +0.27 | 0.604 |
| Two-NN ID (text)               | −0.09 | 0.872 | +0.19 | 0.723 |

**Table 4.** Correlation of each geometric probe with the hard compositional score $H$, across $n = 6$ cells. Bold rows have $\lvert \rho_S \rvert > 0.8$ and $p < 0.05$ on the Spearman test.

The text-side RankMe is the single strongest predictor of $H$, with Spearman $\rho_S = +0.94$ and Pearson $r_P = +0.92$, both significant at $p < 0.01$. The modality cosine and angle are next, with $\lvert \rho_S \rvert = 0.89$ and $p < 0.02$. Audio-side RankMe is also positive and significant ($\rho_S = +0.83$, $p = 0.04$), but it is dominated by the text-side measure.

The probes that *do not* correlate are also informative. The Two-NN intrinsic dimension on both sides has $\lvert \rho_S \rvert \approx 0.1$ and $p > 0.8$ — i.e. essentially no signal. The residual SVD energy at $k = 10$ also has no significant signal. The $\ell_2$ modality gap has no signal either — and this is *not* because the modality gap does not matter; it is because the operative geometry of the modality gap is angular, not absolute, as ([Liang et al., 2022](#references)) already pointed out.

So the headline finding is that, on this benchmark, *text-side effective rank* and the *audio–text modality angle* jointly account for nearly all the between-model variance in compositional retrieval success.

## Discussion

### Text-side capacity dominates

The strongest single predictor of compositional success is the effective rank of the text embedding cloud. The interpretation is straightforward. When the text encoder maps the paired caption $c_p$ and the reversed caption $c_r$ into a tightly low-rank subspace, the joint contrastive head has no axis along which to separate them — even if the audio side carries a discriminating signal. The $G$ (audio-to-text) score is then bottlenecked at the text-encoder output, and the hard $H$ collapses with it.

Audio-side RankMe is also positively correlated with $H$ ($\rho_S = +0.83$), but the text-side correlation is both stronger and tighter. This is consistent with the structural observation that CompA pairs differ only in word ordering or attribute binding: the audio side carries the *discriminating signal* whether or not the model can use it, but the *text* side is where the binary choice is encoded most explicitly. Crush the text spread, and you crush the model's ability to choose.

There is a subtle but important corollary here. Many ablation studies on CLAP-style models focus on swapping audio backbones — HTSAT vs. CNN14 vs. AST vs. M2D — while keeping the text encoder fixed (some flavour of BERT). The result above suggests that for compositional retrieval, the more leveraged knob may actually be on the *other* side: the text encoder, its pretraining corpus, its projection head, and the way the contrastive loss interacts with it. This is not an argument that audio encoders do not matter; it is an argument that the current bottleneck for compositional generalisation in pretrained CLAP is not where most of the engineering effort is being spent.

### Modality alignment matters more than the gap norm

The second result — that the modality angle, not the $\ell_2$ gap, is the predictor — sharpens the original modality-gap story. ([Liang et al., 2022](#references)) showed that joint encoders produce two persistent cones in embedding space and argued that the gap is a generic property of the contrastive objective. The natural follow-up question is: does the gap *matter* for downstream performance? My result here is that, for compositional retrieval, the answer is yes — but the operative quantity is angular. Closing the $\ell_2$ gap by rescaling the centroids closer to the origin does nothing if the centroids still point in different directions. Closing the *angle* gives the joint head a more aligned reference frame to operate in, and that translates directly into compositional accuracy.

MGA-CLAP's two simultaneous wins on this axis — highest effective rank and smallest angle — are not a coincidence. The grouped masked-codebook contrastive objective ([Li et al., 2024](#references)) is structurally biased toward both: by aligning frame-level audio tokens with word-level text tokens through a shared discrete codebook, it forces both encoders to project into a vocabulary of common, balanced bins, which spreads the embedding cloud and reduces angular drift between the two modalities.

### Audio-side collapse is not the main bottleneck

All three models produce audio clouds of broadly similar intrinsic dimension ($10$ to $14$) and similar residual singular-value energy fractions after the top components are removed. The audio side does not appear to be where compositional information is lost; the *joint* space, and in particular the *text encoder's* use of it, is.

This is, again, a slightly counter-intuitive conclusion if you read the CompA scores naïvely. The biggest score gap between LAION-CLAP and MGA-CLAP is on $G$ (audio-to-text), which superficially looks like an audio-encoder problem. But the geometric story says no: the audio embeddings are roughly comparable in shape across models; what differs is what the text encoder gives the joint head to compare them against. When the text cloud is collapsed and the modality angle is wide, the same audio embedding has nothing to lock onto, and audio-to-text retrieval degrades. When the text cloud is rich and the angle is narrow, the audio embedding has a target to align with, and the retrieval score rises.

### Implications for CLAP design

The findings argue for objectives that explicitly preserve text-side spread and reduce angular misalignment. Several existing techniques are natural candidates:

- *Entropy or variance regularisers on the embedding cloud*, in the spirit of VICReg ([Bardes et al., 2022](#references)) or Barlow Twins ([Zbontar et al., 2021](#references)), specifically applied to the text branch.
- *Token-level contrastive alignment* in the spirit of MGA-CLAP ([Li et al., 2024](#references)), which I read as one specific way to achieve text-side spread while simultaneously tightening the modality angle.
- *Compositional data augmentation* — generating hard text negatives during training that differ from the positive only by an order or attribute swap, in the spirit of NegCLIP ([Yuksekgonul et al., 2023](#references)) for vision–language.
- *Cross-modal alignment losses* that explicitly penalise the cosine between modality centroids, rather than only ranking matched pairs above unmatched pairs.

The encouraging part of the result is that MGA-CLAP attains the best $H$ on both benchmarks *without any compositional supervision*. Its training data are the same broadly-scoped audio–caption pairs as the other models; the gain comes from a structural inductive bias in the loss. This suggests that compositional generalisation can be improved through architectural and objective changes that affect embedding geometry, without paying the cost of a compositional supervision pipeline.

### What the probes are good for, and what they are not

The probes I evaluate are *diagnostic*, not *causal*. A high text-side RankMe and a small modality angle correlate with high $H$ across the six cells I have, but this is not the same as saying that artificially inflating RankMe would improve $H$. The correlation is consistent with a story in which a well-designed objective produces both better geometry and better compositional behaviour as joint side-effects, rather than geometry causing behaviour.

That said, the probes are still useful in two practical ways. First, they are *cheap*: they require only a forward pass on unlabelled data and a handful of singular-value decompositions, no labels and no benchmark. If you are training a new CLAP variant and you want a same-day signal about whether it is heading in the right direction for compositional retrieval, RankMe(text) and the modality angle are a defensible first look. Second, they *localise* the failure mode. A model with low $H$ and low RankMe(text) has a text-encoder problem; a model with low $H$, high RankMe(text), and a wide modality angle has a joint-alignment problem. These are different fixes.

## Limitations

I evaluate on six (model × benchmark) cells, which is enough to surface large effects with a Spearman test but is not enough to do fine ranking among the probes. The reported $p$-values should be read as suggestive rather than definitive. A larger sweep — more CLAP variants, more benchmarks, perhaps including non-compositional retrieval benchmarks for contrast — would tighten the picture.

I use a single $10$-second audio window with each model's native sample rate. Longer windows or sliding-window aggregation may shift the audio-side probe values, but my central conclusion — that text-side geometry and the modality angle dominate over audio-side geometry — would only strengthen, since the audio-side probes are already the non-discriminating ones.

M2D-CLAP is run with `flat_features=True` to match the released $768$-dim semantic projection. The alternative $3840$-dim stacked variant is not evaluated here; it would be a natural extension.

All results are inference-only. No hyper-parameter search, no fine-tuning, no prompt-engineering. The point is to characterise *released* CLAP checkpoints, not to push the state of the art.

Finally, the geometric probes themselves have known caveats. RankMe is sensitive to the number of samples $N$ relative to the dimension $D$ when $N \ll D$. Two-NN assumes local uniformity and degrades when the manifold has very heterogeneous local density. Both have been validated empirically across a wide range of self-supervised models ([Garrido et al., 2023](#references); [Facco et al., 2017](#references)), but neither is a perfect statistic.

## Conclusion

A geometry-only analysis of three pretrained CLAP models on CompA shows that text-side effective rank and the audio–text modality angle jointly account for almost all between-model variance in compositional retrieval success. Audio-side intrinsic dimension and residual singular-value energy do not. MGA-CLAP wins on both geometric axes and on the hard score $H$ on both benchmarks.

The broader claim I want to push is that compositional failures in current CLAP models are primarily a *text-space anisotropy and cross-modal alignment* problem, not an *audio-encoder collapse* problem. If that claim survives a larger sweep, it has direct consequences for how the next generation of audio–language models should be designed — and it gives us a cheap, label-free probe to track progress without waiting for benchmark numbers.

Geometric probes computed on unlabelled embeddings are, on this evidence, a useful low-cost proxy for compositional CLAP performance. They are not a substitute for benchmarks, but they are a usefully early signal.

---

Cited as:

> Bahuguna, A. (2026). Compositional Retrieval and Embedding Geometry in Pretrained CLAP Models. *arjunbahuguna.github.io*.

## References

[1] Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. ["Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation."](https://arxiv.org/abs/2211.06687) *Proc. ICASSP*, 2023.

[2] Li, Y., Wang, W., Yang, X., et al. ["Advancing Multi-grained Alignment for Contrastive Language-Audio Pre-training."](https://arxiv.org/abs/2408.07919) *Proc. ACM Multimedia*, 2024.

[3] Niizumi, D., Takeuchi, D., Ohishi, Y., Harada, N., & Kashino, K. ["M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representations."](https://arxiv.org/abs/2406.02032) *Proc. Interspeech*, 2024.

[4] Ghosh, S., Kumar, S., Seth, A., Evuru, C. K. R., Tyagi, U., Sakshi, S., Nieto, O., Duraiswami, R., & Manocha, D. ["CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models."](https://arxiv.org/abs/2310.08753) *Proc. ICLR*, 2024.

[5] Garrido, Q., Balestriero, R., Najman, L., & LeCun, Y. ["RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank."](https://arxiv.org/abs/2210.02885) *Proc. ICML*, 2023.

[6] Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. ["Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information."](https://www.nature.com/articles/s41598-017-11873-y) *Scientific Reports*, 7:12140, 2017.

[7] Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. ["Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning."](https://arxiv.org/abs/2203.02053) *Proc. NeurIPS*, 2022.

[8] Radford, A., Kim, J. W., Hallacy, C., et al. ["Learning Transferable Visual Models From Natural Language Supervision."](https://arxiv.org/abs/2103.00020) *Proc. ICML*, 2021.

[9] Chen, K., Du, X., Zhu, B., Ma, Z., Berg-Kirkpatrick, T., & Dubnov, S. ["HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection."](https://arxiv.org/abs/2202.00874) *Proc. ICASSP*, 2022.

[10] Liu, Y., Ott, M., Goyal, N., et al. ["RoBERTa: A Robustly Optimized BERT Pretraining Approach."](https://arxiv.org/abs/1907.11692) *arXiv preprint*, 2019.

[11] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."](https://arxiv.org/abs/1810.04805) *Proc. NAACL*, 2019.

[12] Niizumi, D., Takeuchi, D., Ohishi, Y., Harada, N., & Kashino, K. ["Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input."](https://arxiv.org/abs/2210.14648) *Proc. ICASSP*, 2023.

[13] Zhao, T., Zhang, T., Zhu, M., Shen, H., Lee, K., Lu, X., & Yin, J. ["VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations."](https://arxiv.org/abs/2207.00221) *arXiv preprint*, 2022.

[14] Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., & Zou, J. ["When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?"](https://arxiv.org/abs/2210.01936) *Proc. ICLR*, 2023.

[15] Hsieh, C.-Y., Zhang, J., Ma, Z., Kembhavi, A., & Krishna, R. ["SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality."](https://arxiv.org/abs/2306.14610) *Proc. NeurIPS*, 2023.

[16] Bardes, A., Ponce, J., & LeCun, Y. ["VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning."](https://arxiv.org/abs/2105.04906) *Proc. ICLR*, 2022.

[17] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. ["Barlow Twins: Self-Supervised Learning via Redundancy Reduction."](https://arxiv.org/abs/2103.03230) *Proc. ICML*, 2021.
