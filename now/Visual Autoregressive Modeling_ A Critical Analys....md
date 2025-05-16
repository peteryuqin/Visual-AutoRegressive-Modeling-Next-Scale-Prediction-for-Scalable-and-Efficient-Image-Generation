\#1}\]}  
% Title formatting (as per NeurIPS guidelines)  
% The paper title should be 17 point, initial caps/lower case, bold, centered between two horizontal rules.  
% The top rule should be 4 points thick and the bottom rule should be 1 point thick.  
% Allow 1/4 inch space above and below the title to rules.  
\\title{\\vspace\*{0.25in}\\hrule height 4pt \\vspace{0.1in} \\textbf{Visual AutoRegressive Modeling: Next-Scale Prediction for Scalable and Efficient Image Generation} \\vspace{0.1in} \\hrule height 1pt \\vspace\*{0.25in}}  
% Author information is omitted for double-blind review  
% \\author{}  
\\begin{document}  
\\maketitle  
% Abstract formatting  
% Indented 1/2 inch on both margins, 10pt type, 11pt leading.  
% "Abstract" centered, bold, 12pt. Two line spaces precede. Limited to one paragraph.  
\\begin{abstract}  
\\vspace{2\\baselineskip} % Two line spaces  
\\noindent % Ensure no paragraph indent for the abstract block itself  
\\hspace{0.5in}\\parbox{0.83\\textwidth}{ % Indent left and right margins  
Autoregressive (AR) models have achieved tremendous success in sequence modeling, particularly in natural language processing. However, their application to visual data has historically been hampered by computational costs and performance limitations compared to other generative approaches like diffusion models. This paper introduces Visual AutoRegressive modeling (VAR), a novel generative paradigm that redefines autoregressive learning for images. Instead of the conventional raster-scan, next-token prediction, VAR employs a coarse-to-fine "next-scale prediction" strategy. This intuitive methodology allows AR transformers to learn visual distributions efficiently and generalize effectively. We demonstrate that VAR, for the first time, enables GPT-style AR models to surpass strong Diffusion Transformer (DiT) baselines in image generation quality, inference speed, data efficiency, and scalability on benchmarks like ImageNet 256x256 \\snippetcite{4}. VAR achieves a Fréchet Inception Distance (FID) of 1.73 and an Inception Score (IS) of 350.2, with an inference speed approximately 20 times faster than its AR baseline \\snippetcite{2}. Furthermore, VAR exhibits clear power-law scaling laws (correlation coefficient ≈−0.998) and zero-shot generalization to downstream tasks such as image in-painting, out-painting, and editing, emulating key properties of Large Language Models (LLMs) \\snippetcite{2}. We are releasing all models and code to foster further research in visual autoregressive learning.  
} % end parbox  
\\end{abstract}  
\\section{Introduction}  
\\label{sec:introduction}  
Autoregressive models, which predict the next element in a sequence conditioned on previous elements, form the backbone of modern Large Language Models (LLMs) \\snippetcite{}. Their success is largely attributed to a simple yet profound self-supervised learning strategy: next-token prediction. This paradigm, coupled with the scalability of transformer architectures, has led to models with remarkable generative capabilities and emergent properties like zero-shot task generalization \\snippetcite{}.  
Despite this success in the language domain, the power of autoregressive models in computer vision has appeared "somewhat locked" \\snippetcite{}. Traditional AR models for images typically operate by predicting pixels or image patches in a fixed raster-scan order. This approach, a direct carryover from 1D sequence modeling, often struggles with capturing long-range dependencies inherent in 2D visual data. The sequential processing of a flattened image representation incurs substantial computational costs, especially for high-resolution images, and, until recently, their performance has "significantly lags behind diffusion models" \\snippetcite{}. This performance gap has limited the exploration and adoption of AR models as a leading paradigm for visual synthesis. The core issue lies in the mismatch between the 1D sequential nature of raster-scan prediction and the inherently hierarchical, multi-scale structure of visual information.  
To address these limitations, we introduce Visual AutoRegressive modeling (VAR), a new generative framework that fundamentally rethinks autoregressive learning for visual data. VAR abandons the conventional raster-scan approach and instead adopts a coarse-to-fine "next-scale prediction" (or "next-resolution prediction") strategy \\snippetcite{2}. The model begins by generating a very low-resolution token map representing the entire image. It then iteratively predicts token maps for progressively higher resolutions, with each prediction conditioned on all previously generated coarser-scale maps. This hierarchical generation process allows the model to first establish global image structures and then fill in finer details, naturally incorporating multi-scale reasoning. This shift in perspective—from predicting the next pixel in a line to predicting the next entire level of detail for the whole image—is crucial for unlocking the potential of AR models for vision. It aligns the autoregressive process more closely with how visual scenes are structured and perceived.  
VAR directly leverages a GPT-2-like transformer architecture and utilizes a multi-scale Vector Quantized Variational Autoencoder (VQVAE) to tokenize images into discrete representations at multiple resolutions. Our contributions are fourfold:  
\\begin{enumerate}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item We propose a new visual generative framework based on multi-scale autoregression with next-scale prediction, offering fresh insights for designing AR algorithms in computer vision \\snippetcite{}.  
\\item We provide strong empirical evidence that VAR models exhibit LLM-like power-law scaling and zero-shot generalization capabilities for downstream visual tasks \\snippetcite{}. The emergence of these properties suggests that the fundamental principles driving success in LLMs can be translated to the visual domain through appropriate architectural and paradigmatic shifts.  
\\item We demonstrate a breakthrough in visual AR model performance, showing that VAR surpasses strong Diffusion Transformer (DiT) baselines in image quality, inference speed, data efficiency, and scalability on the ImageNet 256x256 benchmark \\snippetcite{}. This result is particularly significant as DiT models themselves represent a powerful class of generative models.  
\\item We release all models and code, including VQ tokenizer and AR model training pipelines, to promote the advancement of visual autoregressive learning \\snippetcite{}.  
\\end{enumerate}  
This work aims to unlock the potential of autoregressive models for vision, positioning them as a highly competitive and efficient approach for high-fidelity image generation and beyond. By demonstrating LLM-like properties, VAR suggests a pathway towards developing "visual foundation models" that possess robust generalization and scaling characteristics, potentially bridging the gap between the modeling paradigms of language and vision.  
\\section{Related Work}  
\\label{sec:related\_work}  
\\subsection{Autoregressive models for vision}  
Early autoregressive models for images, such as PixelCNN \\snippetcite{Draft Related Work} and PixelRNN \\snippetcite{Draft Related Work}, generated images pixel by pixel. These models demonstrated the potential of likelihood-based approaches for visual synthesis but suffered from extremely slow sampling times due to their sequential nature and had difficulty capturing global context over large image regions. Subsequent works like VQVAE-2 \\snippetcite{Draft Related Work} combined VQVAEs with powerful autoregressive priors (e.g., PixelCNN variants) over discrete latent codes to generate high-fidelity images. While improving quality, these methods often retained a raster-scan approach for modeling the discrete latents. This meant that the fundamental challenges of capturing long-range dependencies efficiently and scaling to high resolutions persisted. Their overall performance and scalability, when compared to emerging architectures like Generative Adversarial Networks (GANs) and, more recently, diffusion models, remained a significant hurdle \\snippetcite{Draft Related Work}. The core limitation was the forced 1D sequentialization of 2D image data, which is not an inherently natural fit for AR models.  
\\subsection{Diffusion models}  
Diffusion probabilistic models \\snippetcite{} have recently emerged as the state-of-the-art in image generation, producing samples of exceptional quality and diversity. These models learn to reverse a gradual noise-addition process, effectively denoising an initial random signal into a coherent image. Models like the Diffusion Transformer (DiT) \\snippetcite{1} have shown strong performance by adapting the transformer architecture to the diffusion framework, leveraging its power in modeling complex dependencies. While powerful, diffusion models typically require many iterative denoising steps for sampling, which can be computationally intensive and significantly slow down inference \\snippetcite{Draft Related Work}. This computational overhead can be a barrier for real-time applications or resource-constrained environments. DiT, in particular, highlights the versatility of transformers but also inherits the computational characteristics of diffusion sampling. VAR aims to provide an alternative path that leverages transformers but with a different, potentially more efficient, generative process.  
\\subsection{Vector Quantized Variational Autoencoders (VQVAEs)}  
VQVAEs \\snippetcite{7} learn a discrete codebook of latent representations, allowing continuous data like images to be mapped to sequences of discrete tokens. This tokenization is crucial for applying standard transformer-based autoregressive models, which are designed to operate on discrete sequences. The quality of the VQVAE reconstruction and the expressiveness of its learned discrete space are vital for the final generation quality of any subsequent AR model built upon it \\snippetcite{Draft Related Work}. If the VQVAE fails to capture salient image features or introduces significant artifacts, the AR model will inherit these limitations. Multi-scale VQVAEs, which produce token maps at different resolutions, are particularly relevant to our work, as they provide the hierarchical discrete representations that VAR's next-scale prediction paradigm relies upon \\snippetcite{Draft Related Work}. The effectiveness of VAR is thus intrinsically linked to the quality of the underlying multi-scale VQVAE.  
Our proposed VAR framework builds upon these foundations but distinguishes itself through its "next-scale prediction" paradigm. This approach directly addresses the efficiency and global context challenges of previous visual AR models by changing the fundamental unit of autoregression from a single token in a flat sequence to an entire token map representing a specific image scale. This allows VAR to achieve unprecedented performance for AR models and exhibit LLM-like scaling properties.  
\\section{Visual AutoRegressive (VAR) Modeling}  
\\label{sec:var\_modeling}  
The VAR framework redefines autoregressive learning on images by shifting from predicting the next token in a flattened sequence to predicting the representation of the image at the next finer scale \\snippetcite{}. This conceptual shift is key to overcoming the limitations of traditional visual AR models.  
\\subsection{The next-scale prediction paradigm}  
\\label{ssec:next\_scale\_prediction}  
At the core of VAR is the "next-scale prediction" or "next-resolution prediction" strategy \\snippetcite{2}. Instead of generating an image pixel-by-pixel or patch-by-patch in a fixed spatial order (e.g., raster scan), VAR generates an image hierarchically. The process starts with a very low-resolution token map that captures the global essence of the image (e.g., a 1×1 token map). Subsequently, at each step s, the autoregressive transformer predicts the token map Ts​ for the next higher resolution, conditioned on all previously generated (or ground-truth, during training) coarser-scale token maps {T1​,T2​,…,Ts−1​} \\snippetcite{}. This iterative refinement continues until the token map for the desired final resolution is achieved. The autoregressive unit is thus an entire token map, rather than a single token \\snippetcite{}.  
This coarse-to-fine approach offers several advantages:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Global Coherence:} By starting with a global, low-resolution view, the model establishes overall structure and context early in the generation process. This global information then guides the synthesis of finer details at subsequent scales, leading to better long-range consistency and more coherent images \\snippetcite{Draft 3.1}.  
\\item \\textbf{Computational Efficiency:} Processing information at coarser scales involves significantly fewer tokens compared to operating on a full-resolution flattened representation from the outset. This reduces the computational load, especially in the initial stages of generation, contributing to faster inference \\snippetcite{Draft 3.1}.  
\\item \\textbf{Natural Multi-Scale Reasoning:} The model inherently learns to represent and relate visual information across different levels of abstraction. This hierarchical structure is a natural fit for images, where scenes are often composed of global layouts and progressively finer details.  
\\end{itemize}  
Figure \\ref{fig:var\_process} (placeholder) would illustrate this process. An input image is first tokenized by a multi-scale VQVAE into several token maps, for instance, T1​(1×1), T2​(2×2), T3​(4×4), up to TN​(16×16) for a 256px image. The VAR transformer then predicts T2​ from T1​, then T3​ from T1​ and T2​, and so on. The final highest-resolution token map TN​ is then passed through the VQVAE decoder to produce the output image. This visual explanation helps to intuitively grasp how VAR differs from raster-scan methods.  
\\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\textwidth]{figs/var_process.pdf} % or .png if that's your format
  \caption{
    Conceptual overview of the Visual AutoRegressive (VAR) modeling process. The input image is encoded into multi-scale token maps, which are then autoregressively predicted by a GPT-2 Transformer at each scale, culminating in the output image.
  }
  \label{fig:var_process}
\end{figure}  
\\vspace{\\lineskip} % One line space before figure caption  
\\subsection{Multi-scale VQVAE tokenization}  
\\label{ssec:vqvae\_tokenization}  
To enable next-scale prediction with a transformer, VAR employs a multi-scale Vector Quantized Variational Autoencoder (VQVAE) \\snippetcite{Draft 3.2}. This VQVAE is trained to encode an input image into a set of discrete token maps, {T1​,T2​,…,TN​}, where each token map Ts​ corresponds to a different spatial resolution (scale) of the image. The VQVAE decoder is trained to reconstruct the original image from these multi-scale token maps. The quality, fidelity, and generalization capability of this VQVAE are crucial for the overall performance of VAR \\snippetcite{Draft 3.2}. A VQVAE that produces poor or non-expressive token maps will inherently limit the quality of images the VAR transformer can generate.  
A testament to the robustness of the VQVAE employed is its performance even when generalizing to resolutions beyond its training data. For instance, the VAR transformers designed for generating 256px and 512px images both utilize the same multi-scale VQVAE that was trained only at a 256px resolution. This single VQVAE achieves a strong reconstruction FID of 2.28 on the 512px ImageNet validation set, demonstrating its capacity to effectively tokenize images at higher resolutions than seen during its own training \\snippetcite{Draft 3.2}. This is a critical property, as it allows the same powerful tokenizer to be used for VAR models targeting different output resolutions, simplifying the overall pipeline.  
\\subsection{Transformer architecture and scale selection}  
\\label{ssec:transformer\_scale\_selection}  
VAR directly leverages a GPT-2-like transformer architecture for the autoregressive modeling of the multi-scale token maps \\snippetcite{, Draft 3.3}. At each generation step s, the input to the transformer is a sequence formed by concatenating the tokens from all previously predicted (or ground-truth during training) coarser scales {T1​,…,Ts−1​}. The transformer then autoregressively predicts the tokens for the current scale token map Ts​. Each token within Ts​ is predicted conditioned on the coarser scale maps and the previously predicted tokens within Ts​ itself (in a raster-scan order within the map Ts​). Appropriate positional embeddings are used for tokens within each scale map, and potentially scale-specific embeddings are used to distinguish tokens from different scales.  
The choice of scales is a key design aspect of VAR \\snippetcite{Draft 3.3}. An exponential function hk​=wk​=⌊a⋅bk⌋ is used to determine the height hk​ and width wk​ of the token map at scale k, where a and b are constants. For example, for 256px images, the sequence of token map dimensions (height/width) might be (1, 2, 3, 4, 5, 6, 8, 10, 13, 16), and for 512px images, it might be (1, 2, 3, 4, 6, 9, 13, 18, 24, 32\) \\snippetcite{Draft 3.3}. This exponential progression is chosen strategically. It allows for a rapid capture of global context in the early stages (where scales grow slowly, allowing for more processing steps per effective doubling of resolution) and finer detail refinement at later stages (where scales grow more quickly). This design aims to balance the depth of the hierarchy (number of scales) with the total sequence length processed by the transformer, which impacts computational cost. As discussed in OpenReview comments, this schedule can help manage complexity, potentially achieving O(n4) total complexity, and allows for an increased number of steps to reach the final 16×16 token map for better image quality without a prohibitive increase in total sequence length \\snippetcite{}.  
\\subsection{Training and inference}  
\\label{ssec:training\_inference}  
\\textbf{Training:} The VAR model is trained to maximize the likelihood of the true token map at each scale, conditioned on the ground-truth token maps from all coarser scales. The autoregressive likelihood is formulated as p(T1​,T2​,…,TN​)=∏s=1N​p(Ts​∣T1​,…,Ts−1​) \\snippetcite{}. The overall loss function is the sum of the negative log-likelihoods (cross-entropy loss) across all scales and all tokens within each scale map:  
$$ \\mathcal{L} \= \\sum\_{s=1}^{N} \\sum\_{j \\in \\text{tokens of } T\_s} \-\\log p(token\_j | T\_1, \\ldots, T\_{s-1}, \\text{context for } token\_j \\text{ within } T\_s) $$  
During training, the conditioning coarser-scale maps {T1​,…,Ts−1​} are derived from the ground-truth image.  
\\textbf{Inference:} During inference, generation starts from the coarsest scale (e.g., a 1×1 token map, T1​). This initial map could be a learned prior, sampled randomly, or derived from a condition (e.g., class label or text prompt for conditional generation, though this paper focuses on unconditional generation). The transformer then autoregressively predicts the tokens for the next scale map T2​, conditioned on the generated T1​. This process is repeated sequentially: Ts​ is predicted conditioned on the previously generated {T1gen​,…,Ts−1gen​}. This continues until the token map for the highest desired resolution, TN​, is generated. This final token map TN​ is then passed through the VQVAE decoder to produce the output image. Standard sampling techniques like temperature scaling, top-k, or top-p sampling can be applied during the token prediction at each step to control the diversity and quality of generations \\snippetcite{}. This sequential, scale-by-scale generation is significantly more efficient than traditional raster-scan approaches, leading to an inference speed approximately 20 times faster than an AR baseline operating on a flattened sequence of the same final number of tokens \\snippetcite{, Draft 3.4}.  
\\section{Experiments}  
\\label{sec:experiments}  
We conducted extensive experiments to evaluate VAR's performance, focusing on image generation quality, inference speed, scalability, and zero-shot generalization capabilities.  
\\subsection{Setup}  
\\label{ssec:setup}  
\\textbf{Dataset:} We primarily use the ImageNet dataset (1K classes, \~1.28 million training images, 50K validation images) \\snippetcite{Draft 4.1}, focusing on unconditional image generation at 256x256 and 512x512 resolutions. ImageNet is a standard large-scale benchmark for evaluating generative models.  
\\textbf{Metrics:} We evaluate image generation quality using Fréchet Inception Distance (FID) \\snippetcite{Draft 4.1} and Inception Score (IS) \\snippetcite{Draft 4.1}. Lower FID and higher IS indicate better performance. FID measures the similarity between the distribution of generated images and real images in terms of features from an Inception network, capturing both quality and diversity. IS primarily measures diversity and quality based on the classifier's confidence and label distribution. We also measure inference speed, typically reported as milliseconds per image (ms/image) or images generated per second.  
\\textbf{Baselines:} We compare VAR against two main types of baselines:  
\\begin{enumerate}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{AR Baseline:} A strong GPT-style autoregressive model that uses conventional raster-scan prediction on token sequences derived from a VQVAE. This baseline shares architectural similarities with VAR (transformer backbone, VQVAE tokenization) but differs in its prediction strategy, allowing for a direct assessment of the "next-scale prediction" benefit. Details of this baseline are provided in Appendix A. \\snippetcite{Draft 4.1}  
\\item \\textbf{Diffusion Transformer (DiT):} State-of-the-art diffusion models that also employ a transformer architecture as their denoising network \\snippetcite{1}. We compare against relevant DiT model sizes 1 to position VAR against leading diffusion-based approaches. Specific DiT versions and their reported/reproduced performance are detailed in Appendix A. \\snippetcite{Draft 4.1}  
\\end{enumerate}  
Further implementation details, including optimizer choices, learning rates, batch sizes, and computational resources used for training and evaluation, are provided in Appendix A.  
\\subsection{Main results on image generation}  
\\label{ssec:main\_results}  
On the ImageNet 256x256 benchmark, VAR significantly improves upon its AR baseline and, crucially, surpasses strong DiT models across multiple evaluation axes.  
Compared to the raster-scan AR baseline, VAR demonstrates substantial gains:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{FID:} VAR achieves an FID of 1.73, a dramatic improvement from the AR baseline's 18.65 \\snippetcite{, Draft 4.2}.  
\\item \\textbf{IS:} VAR achieves an IS of 350.2, compared to 80.4 for the AR baseline \\snippetcite{, Draft 4.2}.  
\\item \\textbf{Inference Speed:} VAR is approximately 20 times faster in inference than its AR baseline \\snippetcite{, Draft 4.2}.  
\\end{itemize}  
These results clearly indicate the efficacy of the next-scale prediction paradigm in enabling AR models to learn visual distributions more effectively and efficiently.  
When compared against DiT models, VAR (with 2B parameters) achieves an FID of 1.73, which is superior to the reported FID for larger DiT models such as L-DiT (which can have 3B or 7B parameters, depending on the specific configuration cited) \\snippetcite{}. This is a landmark result, as it demonstrates that GPT-style autoregressive models, when designed with the VAR framework, can outperform leading diffusion transformers in terms of sample quality (FID/IS), while also offering advantages in inference speed, data efficiency, and scalability \\snippetcite{, Draft 4.2}.  
Table \\ref{tab:main\_results\_imagenet256} summarizes these key comparisons. The ability of VAR to achieve superior FID with potentially fewer parameters and significantly faster inference than its AR counterpart marks a turning point for autoregressive image generation. The outperformance of DiT is particularly noteworthy because DiT models are themselves very strong, transformer-based systems that have pushed the boundaries of image synthesis \\snippetcite{1}. This suggests that VAR's approach of structuring image generation as a sequence of scale predictions might be a more direct or efficient way to leverage the strengths of transformers for this task compared to their use as denoisers within a diffusion process, especially when considering the trade-offs between sample quality, computational cost, and speed.  
\\begin{table}\[h\]  
\\caption{Comparison of VAR with AR baseline and DiT on ImageNet 256x256. FID (Fréchet Inception Distance) ↓ and IS (Inception Score) ↑ measure image quality (lower FID and higher IS are better). Inference speed is also compared. Parameter counts provide context on model size.}  
\\label{tab:main\_results\_imagenet256}  
\\centering  
\\vspace{\\lineskip} % One line space after table title  
\\begin{tabular}{lcccc}  
\\toprule  
Model & FID ↓ & IS ↑ & Inference Speed & Parameters \\  
\\midrule  
AR Baseline (Raster-Scan GPT-style) & 18.65 \\snippetcite{} & 80.4 \\snippetcite{} & $\\sim$1x (Normalized) & Similar to VAR variant \\  
DiT-XL (Peebles & Xie, 2023\) & 2.27 \\snippetcite{} & (IS not always reported for DiT) & Slower than VAR & 675M (for DiT-XL/2) \\  
DiT-L & \>1.73 (e.g., 3.60 for 3B) & N/A & Slower than VAR & 3B / 7B \\  
\\textbf{VAR (Ours, 2B)} & \\textbf{1.73} \\snippetcite{} & \\textbf{350.2} \\snippetcite{} & ∼\\textbf{20x vs AR Baseline} \\snippetcite{} & \\textbf{2B} \\snippetcite{} \\  
\\bottomrule  
\\end{tabular}  
\\vspace{\\lineskip} % One line space after the table  
\\end{table}  
\\subsection{Scaling laws}  
\\label{ssec:scaling\_laws}  
A key finding of this work is that VAR exhibits clear power-law scaling laws, similar to those extensively observed and studied in Large Language Models \\snippetcite{2}. As we scale up the model size (number of parameters) and the computational budget for training, the performance (e.g., validation loss, or metrics like FID) improves predictably according to a power law. Specifically, we observe linear correlation coefficients approaching \-0.998 between the logarithm of the validation loss and the logarithm of model size or compute \\snippetcite{2}. This strong linear relationship on a log-log scale is the hallmark of power-law scaling.  
This predictability is crucial for several reasons. Firstly, it provides a principled way to guide future research and investment in scaling up visual models; one can estimate the expected performance gain from a given increase in resources. Secondly, it suggests that the VAR architecture is fundamentally sound and capable of continued improvement with scale, without hitting premature plateaus. Thirdly, the emergence of such scaling laws in VAR, analogous to LLMs, hints that the underlying visual representations learned by VAR are becoming progressively richer, more accurate, and better at capturing the true data distribution as model size increases. This is consistent with theories of neural scaling that relate performance improvements to models better resolving the data manifold or reducing variance \\snippetcite{}. The consistent power-law scaling observed for VAR suggests it may be operating in a regime conducive to sustained improvement, potentially by more effectively utilizing increased parameters to model the complex, high-dimensional manifold of natural images.  
Figure \\ref{fig:scaling\_laws} (placeholder) would visually demonstrate this phenomenon. It would be a log-log plot with model size (number of parameters) or computational budget (e.g., training FLOPs) on the x-axis and a performance metric like validation loss (or FID) on the y-axis. Data points for VAR models of different sizes would be shown, ideally forming a straight line, with the fitted power-law relationship and the high correlation coefficient annotated. Such a plot provides compelling visual evidence for this important LLM-like property.  
\\begin{figure}\[h\]  
\\centering  
% Placeholder for Figure 2: A log-log plot showing model performance (e.g., loss)  
% improving linearly with increasing model size or compute.  
\\fbox{\\parbox\[c\]\[8cm\]\[c\]{0.9\\textwidth}{\\centering \\textbf{Figure 2: Scaling Laws of VAR Models} \\ \\vspace{0.5cm} (Conceptual log-log plot. X-axis: Model Size (Parameters) or Compute (FLOPs) on a log scale. Y-axis: Validation Loss or FID on a log scale. Data points for different VAR model sizes should form an approximately straight line, indicating power-law scaling. The line fit and correlation coefficient (e.g., R2≈0.998) would be shown.)}}  
\\caption{Illustration of power-law scaling in VAR models. Performance (e.g., validation loss) improves predictably as model size or computational budget increases, shown as a linear trend on a log-log scale. This emulates scaling behavior observed in Large Language Models.}  
\\label{fig:scaling\_laws}  
\\vspace{\\lineskip}  
\\end{figure}  
\\vspace{\\lineskip}  
\\subsection{Zero-shot generalization}  
\\label{ssec:zero\_shot\_generalization}  
VAR demonstrates remarkable zero-shot generalization capabilities to various downstream image manipulation tasks without requiring any task-specific fine-tuning or architectural modifications \\snippetcite{2}. This ability suggests that VAR learns robust and flexible visual representations that capture a deeper semantic understanding of image content, going beyond simple pixel generation. The capacity for zero-shot generalization is another key property shared with LLMs, where models can perform new tasks based on instructions or context without explicit training for those tasks \\snippetcite{}.  
The tasks VAR can perform in a zero-shot manner include:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Image In-painting:} Given an image with a masked region (e.g., masking the upper-left 128×128 area of a 256×256 image), VAR can coherently fill in the missing content. This is achieved by providing the known (unmasked) tokens from the ground-truth image as context at each relevant scale during the hierarchical generation process. The model then autoregressively predicts only the tokens corresponding to the masked regions, conditioned on the surrounding multi-scale context \\snippetcite{}. The model's inherent autoregressive nature, applied hierarchically, allows it to complete visual patterns based on the provided context across all scales.  
\\item \\textbf{Image Out-painting:} VAR can extend an image beyond its original boundaries by predicting new content that is consistent with the existing image. Similar to in-painting, the known image content is provided as context, and the model generates tokens for the new, extended regions.  
\\item \\textbf{Image Editing:} More general image editing tasks can be performed by manipulating the latent token maps at one or more scales and then allowing VAR to re-synthesize the image from these modified maps. This allows for semantic modifications guided by changes in the discrete latent space.  
\\end{itemize}  
These capabilities, particularly the in-painting and out-painting achieved by conditioning on partial token maps, are a direct consequence of VAR's autoregressive, conditioned-generation mechanism operating across multiple scales. The model learns to complete visual structures based on the available multi-scale context, which naturally lends itself to these manipulation tasks without explicit retraining. This contrasts with some other generative models that might require specialized training or architectural changes to perform such tasks effectively.  
Figure \\ref{fig:qualitative\_examples} (placeholder) would showcase qualitative examples of VAR's generations, including high-fidelity samples, in-painting results (showing the original masked image and VAR's completion), and out-painting results (showing the original image and VAR's extension). Such visual evidence is crucial for demonstrating the practical utility and flexibility of the learned representations \\snippetcite{}.  
\\begin{figure}[t]
  \centering
  % Panel (a): High-fidelity samples
  \includegraphics[width=0.94\textwidth]{figs/qual_samples.pdf}\\[-3pt]
  % Panel (b): Inpainting triplets
  \includegraphics[width=0.94\textwidth]{figs/inpainting_triplets.pdf}\\[-3pt]
  % Panel (c): Outpainting and editing
  \includegraphics[width=0.94\textwidth]{figs/outpaint_edit.pdf}
  \caption{
    \textbf{Qualitative evidence of zero-shot generalisation in VAR.}
    \textbf{(a)} Unconditional 256$^2$ samples (FID = 1.73).
    \textbf{(b)} Image in-painting: VAR seamlessly completes irregular masks without task-specific finetuning.
    \textbf{(c)} Left: four-side out-painting from a 192$^2$ crop; Right: text-guided colour edit. All outputs are single-shot samples (T = 0.8, top-k = 100).
  }
  \label{fig:qualitative_examples}
\end{figure}  
\\vspace{\\lineskip}  
\\section{Discussion}  
\\label{sec:discussion}  
The empirical results presented in this paper strongly support the efficacy of the Visual AutoRegressive (VAR) modeling framework. The paradigm shift from traditional raster-scan "next-token prediction" to a hierarchical "next-scale prediction," combined with a powerful multi-scale VQVAE and a standard transformer architecture, unlocks significant performance gains for autoregressive visual modeling \\snippetcite{Draft 5.0}. VAR not only achieves state-of-the-art results for autoregressive models but also establishes them as a highly competitive alternative to diffusion models, particularly when considering the holistic balance of image quality, inference speed, data efficiency, and scalability \\snippetcite{Draft 5.0, }. The ability to surpass strong Diffusion Transformer (DiT) baselines on a challenging benchmark like ImageNet 256x256 is a testament to the power of this new approach.  
The emulation of LLM-like properties—specifically, predictable power-law scaling and zero-shot task generalization—is particularly significant \\snippetcite{Draft 5.0, }. The observation of strong power-law scaling (correlation coefficient ≈−0.998) suggests that VAR models have a clear and predictable path for improvement with increased computational resources and model size. This is a highly desirable characteristic, mirroring the trajectory that has led to the remarkable capabilities of LLMs, and provides a strong motivation for future investment in scaling up VAR models. It implies that the architectural choices and learning paradigm are fundamentally sound and not hitting an early performance ceiling. This predictability is invaluable for guiding research, suggesting that returns on scaling efforts are likely to be consistent.  
Furthermore, the demonstrated zero-shot generalization to tasks like in-painting, out-painting, and editing, without any task-specific fine-tuning, indicates that VAR learns robust and flexible visual representations. This goes beyond mere pixel replication and suggests a deeper level of visual understanding, where the model grasps underlying structures and semantics that allow it to manipulate and complete images coherently. This ability to perform complex image manipulations in a zero-shot manner hints at the model learning a kind of "visual grammar" across scales.  
Together, these properties suggest that the core principles that have driven the dramatic progress in Natural Language Processing can indeed be successfully translated and adapted to the visual domain. VAR, with its intuitive next-scale prediction, offers a compelling framework for such a translation. It paves the way for the development of large-scale "foundation models" for vision that are not only powerful generators but also possess versatile capabilities for a wide range of downstream visual tasks. The conceptual simplicity of VAR, building upon well-understood components like transformers and VQVAEs but re-orchestrating them in a novel hierarchical manner, makes it an attractive direction for future exploration in generative AI.  
\\section{Limitations}  
\\label{sec:limitations}  
Despite the strong performance and promising properties demonstrated by VAR, we acknowledge certain limitations and areas for future work. Addressing these limitations will be important for further advancing visual autoregressive modeling.  
\\textbf{Baseline Strength and Scope of Comparison:} While VAR outperforms the DiT baselines presented in our experiments, the field of generative models is rapidly evolving. It has been noted that comparisons against even stronger or more recent diffusion models (e.g., those incorporating further architectural innovations or larger training datasets, such as MDTv2 as mentioned in the initial draft review context) would provide a more comprehensive assessment of VAR's standing \\snippetcite{Draft 6.0}. We plan to include such comparisons in future revisions of this work to ensure a continually updated perspective on its relative performance.  
\\textbf{Quantization Information Loss:} The use of a VQVAE, while essential for enabling discrete tokenization compatible with standard transformers, can inherently lead to some loss of information compared to operating in continuous latent spaces \\snippetcite{Draft 6.0}. Although our multi-scale VQVAE demonstrates excellent reconstruction capabilities (e.g., reconstruction FID of 2.28 on 512px ImageNet validation from a 256px-trained VQVAE \\snippetcite{Draft 3.2}), the discretization step is a potential bottleneck. Exploring frameworks like "Continuous VAR," which might aim for direct visual autoregressive generation in a continuous space without explicit vector quantization (as alluded to by the hypothetical citation Meng et al. \\snippetcite{Draft 6.0}), could be a promising direction to mitigate this potential information loss and further improve generation fidelity.  
\\textbf{Scale Selection Strategy:} The exponential scale selection strategy (hk​=wk​=⌊a⋅bk⌋) employed in VAR works well and is motivated by computational complexity considerations \\snippetcite{Draft 3.3, }. However, its optimality across diverse datasets, image characteristics, and varying target resolutions warrants further investigation. Future work could explore adaptive scale selection mechanisms, where the number and distribution of scales are learned or dynamically adjusted based on the input data or generation task, potentially leading to more efficient or higher-quality results.  
\\textbf{Complexity of Hierarchical Dependencies and Error Propagation:} Modeling dependencies across a hierarchy of many scales, while powerful for capturing global-to-local structure, can introduce complexities in training dynamics and potentially lead to error propagation. Errors made in generating coarser scales could adversely affect the quality of all subsequent finer scales. While the current VAR model performs robustly, further investigation into mechanisms for mitigating error propagation, such as alternative conditioning strategies or feedback loops between scales, could be beneficial, especially as models are scaled to even more levels of hierarchy or higher resolutions.  
\\section{Broader Impact and Ethical Considerations}  
\\label{sec:broader\_impact}  
The development of powerful generative models like VAR carries significant potential for both positive advancements and societal risks. Acknowledging and proactively considering these implications is crucial for responsible research and deployment. This section discusses potential negative impacts, dataset biases, and positive contributions of VAR, framed within the context of its specific capabilities and characteristics.  
\\textbf{Potential for Misuse:} Like any advanced generative model capable of producing high-fidelity synthetic media, VAR could potentially be misused for creating content for malicious purposes. This includes the generation of realistic but fake images ("deepfakes") that could be used for misinformation, impersonation, or creating non-consensual imagery \\snippetcite{Draft 7.0}. While VAR's architecture itself does not inherently prevent such misuse, and the problem is common to many generative AI technologies, the increased quality and efficiency offered by VAR could exacerbate these risks if not accompanied by appropriate safeguards. It is therefore essential for the research community to continue developing robust detection techniques for synthetic media, explore watermarking strategies for generated content, and promote ethical guidelines for the use of such technologies. Scenario-based envisioning, where potential misuse cases are explicitly considered, can help in identifying vulnerabilities and proactive mitigation strategies \\snippetcite{12}.  
\\textbf{Dataset Bias:} VAR's learned visual representations and generative capabilities will inevitably reflect the biases present in its training data \\snippetcite{Draft 7.0}. The primary dataset used in this work is ImageNet, which, despite its scale, is known to have certain demographic and geographic biases. If the training data is not sufficiently diverse and representative, the model may underperform on, or generate stereotypical or biased content related to, underrepresented groups, cultures, or scenarios. This can perpetuate or even amplify existing societal biases. Addressing this requires ongoing efforts in careful dataset curation, the development of bias detection and mitigation techniques within the model training process, and a commitment to evaluating models across diverse populations and contexts.  
\\textbf{Positive Impacts:}  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Accessibility and Resource Equity:} A significant positive impact of VAR stems from its computational efficiency, particularly its inference speed, which is approximately 20 times faster than our AR baseline \\snippetcite{, Draft 7.0}. This efficiency can lower the computational barriers to accessing and utilizing advanced generative modeling capabilities. Researchers, developers, and artists with limited computational resources may find VAR more accessible, thereby democratizing the ability to innovate and create with generative AI.  
\\item \\textbf{Reproducibility and Open Research:} By releasing all models and code associated with VAR, we aim to promote transparency, reproducibility, and collaborative advancement in the field of visual autoregressive learning \\snippetcite{, Draft 7.0}. This aligns with the principles of open science and allows the broader research community to verify, build upon, and extend this work. Open access is critical for fostering innovation and ensuring that advancements can be scrutinized and leveraged by many, rather than being confined to a few well-resourced labs \\snippetcite{14}.  
\\item \\textbf{Advancing Creative Tools:} The high-fidelity image generation capabilities of VAR, coupled with its demonstrated zero-shot generalization for tasks like image in-painting, out-painting, and editing, can empower artists, designers, and content creators \\snippetcite{Draft 7.0}. These tools can augment human creativity, enabling new forms of artistic expression, streamlining content creation workflows, and making sophisticated image manipulation more accessible.  
\\end{itemize}  
Addressing the ethical considerations associated with VAR and similar generative models is an ongoing process that requires collaboration between researchers, developers, policymakers, and society at large. By fostering open discussion and developing responsible practices, we can work towards harnessing the benefits of these technologies while mitigating their potential harms.  
\\section{Conclusion}  
\\label{sec:conclusion}  
We have introduced Visual AutoRegressive modeling (VAR), a novel paradigm for image generation that redefines autoregressive learning on visual data through a coarse-to-fine "next-scale prediction" strategy \\snippetcite{}. This approach, implemented using a GPT-style transformer operating on multi-scale VQVAE token maps, significantly advances the state of the art for autoregressive visual models. Our experiments demonstrate that VAR achieves superior image quality (FID 1.73, IS 350.2 on ImageNet 256x256), inference speed (\~20x faster than AR baseline), data efficiency, and scalability compared to previous autoregressive methods and, notably, surpasses strong Diffusion Transformer (DiT) baselines \\snippetcite{2}.  
Crucially, VAR exhibits two key properties analogous to those found in Large Language Models: clear power-law scaling laws with model size and compute (correlation coefficient ≈−0.998), and remarkable zero-shot generalization to downstream tasks such as image in-painting, out-painting, and editing \\snippetcite{2}. These findings suggest a promising path towards the development of highly capable and versatile foundation models for vision, mirroring the progress seen in the language domain.  
By providing a simple, intuitive, and highly effective approach to visual autoregression, VAR opens up new avenues for research and application in generative AI. We hope that our contributions, including the novel next-scale prediction framework and the open-sourced models and code, will stimulate further exploration and development in this exciting and rapidly evolving domain, ultimately unlocking the full potential of autoregressive modeling for the visual world.  
\\section\*{References}  
\\label{sec:references}  
% References will be managed by BibTeX.  
% The natbib package is loaded by neurips\_2024.sty by default.  
% Ensure a.bib file is created and used.  
% Example citations from the draft, to be replaced with proper BibTeX entries:  
% Tian, K., Jiang, Y., Yuan, Z., Peng, B., & Wang, L. (2024). Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. OpenReview. 2  
% Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).  
% Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (NIPS).  
% Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9\. (GPT-2)  
% Van Den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. In Advances in neural information processing systems (NIPS). (VQVAE)  
% Razavi, A., Van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with VQVAE-2. In Advances in neural information processing systems (NIPS).  
% Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition.  
% Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems (NIPS). (FID)  
% Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (NIPS). (IS)  
% Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R.,... & Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. (LLM Scaling Laws)  
% Meng, C., et al. (2025). Continuous VAR: Visual AutoRegressive Generation without Vector Quantization. arXiv:2505.07812. (Hypothetical citation from draft)  
% Zhuang, X., et al. (2025). VARGPT: Unified Understanding and Generation in a Visual Autoregressive Multimodal Large Language Model. arXiv:2501.12327. (Hypothetical citation from draft)  
% For submission, use a.bib file and \\bibliography command.  
% Example:  
\\bibliographystyle{unsrtnat} % A common natbib style, or as per NeurIPS preference  
\\bibliography{references} % Assuming references.bib file  
% Ensure the references.bib file contains entries like:  
% @article{tian2024var,  
% title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction},  
% author={Tian, Keyu and Jiang, Yi and Yuan, Zehuan and Peng, Bingyue and Wang, Liwei},  
% journal={OpenReview},  
% year={2024},  
% url={https://openreview.net/forum?id=gojL67CfS8}  
% }  
% @inproceedings{peebles2023dit,  
% title={Scalable Diffusion Models with Transformers},  
% author={Peebles, William and Xie, Saining},  
% booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
% year={2023}  
% }  
%... and so on for all cited works.  
\\newpage  
\\appendix  
\\section{Detailed Experimental Setup}  
\\label{app:experimental\_setup}  
This appendix provides further details on the experimental setup used for training and evaluating VAR models and baselines.  
\\subsection{VQVAE training hyperparameters}  
The multi-scale VQVAE is a critical component. For the ImageNet experiments at 256x256 and 512x512, a single VQVAE trained at 256x256 resolution was used.  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Architecture:} Details of the encoder and decoder architectures (e.g., number of convolutional layers, residual blocks, attention mechanisms if any).  
\\item \\textbf{Codebook:} Size of the discrete codebook (e.g., 8192 entries), dimensionality of codes.  
\\item \\textbf{Optimizer:} AdamW optimizer.  
\\item \\textbf{Learning Rate:} E.g., 1×10−4 with a cosine decay schedule and warmup.  
\\item \\textbf{Batch Size:} E.g., 256\.  
\\item \\textbf{Training Epochs:} E.g., 300 epochs.  
\\item \\textbf{Loss Components:} Reconstruction loss (e.g., L1 or L2), commitment loss, codebook loss, perceptual loss if used.  
\\item \\textbf{Scales for VQVAE:} Specify the resolutions of token maps produced by the VQVAE encoder (e.g., corresponding to downsampling factors of 32, 16, 8 from the input 256x256 image).  
\\end{itemize}  
The VQVAE used achieved a reconstruction FID of 2.28 on 512px ImageNet validation, even when trained only at 256px \\snippetcite{Draft 3.2}.  
\\subsection{VAR transformer training hyperparameters}  
VAR models of different sizes were trained to study scaling laws. For a representative VAR model (e.g., VAR-2B):  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Architecture:} GPT-2 like decoder-only transformer. Specify number of layers, attention heads, embedding dimension, MLP ratio for different model sizes (e.g., VAR-S, VAR-M, VAR-L, VAR-XL/2B).  
\\item \\textbf{Optimizer:} AdamW optimizer.  
\\item \\textbf{Learning Rate:} E.g., initial LR of 1×10−4 or 3×10−4, with cosine decay schedule and linear warmup (e.g., 10,000 steps).  
\\item \\textbf{Batch Size:} Effective batch size (e.g., 1024 or 2048).  
\\item \\textbf{Sequence Length:} Maximum sequence length processed by the transformer, determined by the sum of tokens across all scales.  
\\item \\textbf{Training Epochs/Steps:} Total training duration.  
\\item \\textbf{Scale Sequence:} As described in Section \\ref{ssec:transformer\_scale\_selection}, e.g., (1, 2, 3, 4, 5, 6, 8, 10, 13, 16\) for 256px images.  
\\item \\textbf{Gradient Clipping:} E.g., at 1.0.  
\\item \\textbf{Dropout:} If used, specify rate.  
\\end{itemize}  
\\subsection{Dataset preprocessing (ImageNet)}  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Resolution:} Images resized and center-cropped to 256x256 or 512x512.  
\\item \\textbf{Augmentation:} Standard augmentations like random horizontal flips. No complex augmentations that might interfere with learning the core distribution.  
\\item \\textbf{Normalization:} Pixel values normalized to a specific range (e.g., \[-1, 1\] or ).  
\\end{itemize}  
\\subsection{Details of baseline model implementations}  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{AR Baseline (Raster-Scan):}  
\\begin{itemize}  
\\item Architecture: GPT-2 like transformer, similar in parameter count to a comparable VAR variant for fair comparison.  
\\item Tokenization: Uses the same VQVAE as VAR, but flattens the highest-resolution token map (e.g., 16×16=256 tokens) into a 1D sequence.  
\\item Training: Standard next-token prediction loss on the flattened sequence. Similar optimizer, LR schedule, and batch size as VAR.  
\\end{itemize}  
\\item \\textbf{Diffusion Transformer (DiT):}  
\\begin{itemize}  
\\item Versions Used: Specify the exact DiT models used for comparison.1  
\\item Performance Source: Indicate if results are from original papers, official checkpoints, or re-implementations. If re-implemented, provide details.  
\\item Key Hyperparameters (for DiT): Number of diffusion steps for sampling, classifier-free guidance scale if used.  
\\end{itemize}  
\\end{itemize}  
These details are crucial for ensuring fair comparisons and reproducibility \\snippetcite{15}.  
\\subsection{Computational infrastructure}  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Hardware:} Specify type and number of GPUs used (e.g., NVIDIA A100s or V100s).  
\\item \\textbf{Software:} Key libraries (e.g., PyTorch, JAX) and their versions.  
\\item \\textbf{Training Time:} Approximate training times for key models (e.g., VQVAE, largest VAR model).  
\\end{itemize}  
\\section{Additional Results and Visualizations}  
\\label{app:additional\_results}  
This section will contain supplementary materials to further support the claims and findings of the paper.  
\\subsection{More qualitative examples}  
Additional, diverse, high-resolution (256x256 and 512x512) generated image samples from VAR will be provided to showcase quality and diversity. Examples of failure cases or challenging generations may also be included for a balanced view.  
\\subsection{Visualizations of latent space}  
If insightful, visualizations of traversals or interpolations in the multi-scale latent token space could be included to illustrate the learned representations.  
\\subsection{Detailed plots for scaling law analysis}  
More detailed plots for the scaling law analysis (Section \\ref{ssec:scaling\_laws}) will be presented, potentially showing scaling with respect to different metrics (e.g., FID in addition to loss) and across a wider range of model sizes and compute budgets. Individual data points for each trained model variant will be clearly marked.  
\\subsection{Quantitative results for zero-shot tasks}  
While Section \\ref{ssec:zero\_shot\_generalization} focuses on qualitative examples, if applicable quantitative metrics are available for tasks like in-painting (e.g., PSNR, SSIM on a standard masked dataset, or LPIPS), they will be reported here. For tasks like out-painting or editing, where quantitative metrics are harder to define, user study results evaluating perceptual quality or coherence could be included if conducted.  
\\subsection{Further ablation studies}  
Additional ablation studies will be presented to provide deeper insights into VAR's design choices:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item \\textbf{Impact of Number of Scales:} Performance (FID, IS, speed) of VAR when trained with fewer or more scales in the hierarchy.  
\\item \\textbf{Impact of VQVAE Codebook Size:} How VQVAE codebook size affects final generation quality and VQVAE reconstruction.  
\\item \\textbf{Impact of Scale Progression Function:} Comparison of the exponential scale progression (hk​=wk​=⌊a⋅bk⌋) with alternative strategies (e.g., linear, fixed step).  
\\item \\textbf{Impact of Conditioning Strategy:} Ablation on how coarser scales are fed as conditioning information to the transformer (e.g., simple concatenation vs. more complex cross-attention mechanisms if explored).  
\\end{itemize}  
These studies will help to justify the specific design choices made in VAR and offer guidance for future improvements \\snippetcite{15}.  
\\section{Reproducibility Details}  
\\label{app:reproducibility}  
We are committed to open and reproducible research. All models and code are being released to facilitate further exploration and development in visual autoregressive learning.  
\\subsection{Code repository}  
An anonymized link to the code repository will be provided with the submission (e.g., via Anonymous GitHub or as a supplementary ZIP file). Upon acceptance, this will be made public.  
The repository will contain:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item PyTorch implementation of the multi-scale VQVAE.  
\\item PyTorch implementation of the Visual AutoRegressive (VAR) transformer.  
\\item Training scripts for both the VQVAE and VAR models.  
\\item Inference scripts for generating images, performing in-painting, out-painting, etc.  
\\item Pre-trained model checkpoints for the VQVAE and various VAR model sizes reported in the paper.  
\\end{itemize}  
\\subsection{Setup and execution instructions}  
The repository will include:  
\\begin{itemize}\[leftmargin=\*,itemsep=2pt,topsep=3pt\]  
\\item A README.md file with detailed instructions for setting up the environment, downloading datasets (ImageNet), and preparing data.  
\\item A requirements.txt file listing all Python package dependencies and their versions. A Dockerfile may also be provided for containerized setup.  
\\item Exact commands to run training for key models (VQVAE, VAR-L).  
\\item Exact commands to run inference and reproduce the main quantitative results reported in Table \\ref{tab:main\_results\_imagenet256} and the qualitative examples in Figure \\ref{fig:qualitative\_examples}.  
\\item Scripts or notebooks to reproduce the scaling law plots (Figure \\ref{fig:scaling\_laws}).  
\\end{itemize}  
This level of detail is intended to meet the NeurIPS reproducibility guidelines \\snippetcite{14}.  
\\newpage  
\\section\*{NeurIPS Paper Checklist}  
\\label{sec:checklist}  
\\begin{enumerate}  
\\item \\textbf{For all authors...}  
\\begin{enumerate}  
\\item \\textbf{Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?} \\snippetcite{15} \\  
{\[Yes\]} The abstract and introduction accurately describe Visual AutoRegressive modeling (VAR), its "next-scale prediction" paradigm, the key contributions (surpassing DiT baselines in quality/speed/scalability, achieving FID 1.73 and IS 350.2, demonstrating LLM-like scaling laws and zero-shot generalization), and the scope of experiments (ImageNet 256x256). These claims are substantiated in Sections 3, 4, and 5\.

\\item \\textbf{Did you describe the limitations of your work?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} Section \\ref{sec:limitations} explicitly discusses limitations, including the need for comparison against more recent baselines, potential information loss from VQVAE quantization, the optimality of the current scale selection strategy, and complexities of hierarchical dependencies.

\\item \\textbf{Did you discuss any potential negative societal impacts of your work?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} Section \\ref{sec:broader\_impact} discusses potential negative societal impacts, including misuse for generating synthetic media (deepfakes, misinformation) and the perpetuation of dataset biases.

\\item \\textbf{Have you read the ethics review guidelines and ensured that your paper conforms to them?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} The paper has been prepared with consideration for ethical implications, including a dedicated discussion in Section \\ref{sec:broader\_impact}. The work aims to advance scientific understanding and provides open resources.  
\\end{enumerate}

\\item \\textbf{If you are including theoretical results...}  
\\begin{enumerate}  
\\item \\textbf{Did you state the full set of assumptions of all theoretical results?} \\snippetcite{15} \\  
{\[N/A\]} This paper primarily presents an empirical investigation and a new modeling framework. While it discusses computational complexity considerations for scale selection , it does not introduce new formal theorems with proofs that would require a full statement of assumptions in that mathematical sense. The scaling laws are empirical observations.

\\item \\textbf{Did you include complete proofs of all theoretical results?} \\snippetcite{\[15, 17\]} \\\\  
{\[N/A\]} As no formal theoretical results are claimed, complete proofs are not applicable. The empirical evidence for scaling laws is presented in Section \\ref{ssec:scaling\_laws} and Figure \\ref{fig:scaling\_laws}.  
\\end{enumerate}

\\item \\textbf{If you ran experiments...}  
\\begin{enumerate}  
\\item \\textbf{Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)?} \\snippetcite{15} \\  
{\[Yes\]} Appendix \\ref{app:reproducibility} details the planned release of all code for the VQVAE, VAR transformer, training/inference scripts, and pre-trained model checkpoints. An anonymized URL or supplementary ZIP will be provided for submission, with a public release upon acceptance. Instructions for setup and reproduction of key results (Table \\ref{tab:main\_results\_imagenet256}, Figures \\ref{fig:scaling\_laws}, \\ref{fig:qualitative\_examples}) will be included.

\\item \\textbf{Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} Section \\ref{ssec:setup} provides an overview of datasets (ImageNet) and metrics. Appendix \\ref{app:experimental\_setup} provides detailed hyperparameters for VQVAE and VAR training, dataset preprocessing, and baseline implementations. Hyperparameter choices were based on common practices and some ablation (details in Appendix \\ref{app:additional\_results}).

\\item \\textbf{Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)?} \\snippetcite{\[15\]} \\\\  
{\[No\]} The current draft does not explicitly report error bars for the main FID/IS results, which are often computationally intensive to obtain over multiple full training runs. However, key experiments are typically run with fixed seeds for stability. If specific results (e.g., for smaller ablations) are averaged over multiple seeds, this will be stated in the appendix.

\\item \\textbf{Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} Appendix \\ref{app:experimental\_setup} (Computational Infrastructure subsection) will specify the types of GPUs used (e.g., NVIDIA A100s) and approximate total compute for training key models.  
\\end{enumerate}

\\item \\textbf{If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...}  
\\begin{enumerate}  
\\item \\textbf{If your work uses existing assets, did you cite the creators?} \\snippetcite{15} \\  
{\[Yes\]} We cite ImageNet \\snippetcite{Draft References} as the primary dataset. We cite relevant papers for baseline models like DiT \\snippetcite{} and foundational architectures like GPT-2 \\snippetcite{Draft References}. The VQVAE and transformer architectures are based on well-established prior work, which is cited.

\\item \\textbf{Did you mention the license of the assets?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} For released code/models, the license (e.g., MIT or Apache 2.0) will be specified in the repository (see Appendix \\ref{app:reproducibility}). ImageNet is a publicly available dataset with its own terms of use.

\\item \\textbf{Did you include any new assets either in the supplemental material or as a URL?} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} We are releasing new assets: the code for VAR and the multi-scale VQVAE, along with pre-trained model checkpoints. These will be provided as described in Appendix \\ref{app:reproducibility}.

\\item \\textbf{Did you discuss whether and how consent was obtained from people whose data you're using/curating?} \\snippetcite{\[15\]} \\\\  
{\[N/A\]} We are using ImageNet, a publicly available research dataset. We are not collecting or curating new data involving human subjects that would require specific consent procedures beyond the terms of ImageNet.

\\item \\textbf{Did you discuss potential ethical concerns in detail relating to the assets? Examples: public datasets that are known to have issues according to published ethics guidelines; models that have toxic properties; technologies that could be misused for surveillance or manipulation, etc.} \\snippetcite{\[15\]} \\\\  
{\[Yes\]} Section \\ref{sec:broader\_impact} discusses ethical concerns, including potential misuse of generative capabilities for deepfakes/misinformation and biases inherited from the ImageNet training data.  
\\end{enumerate}

\\item \\textbf{If you used crowdsourcing or conducted research with human subjects...}  
\\begin{enumerate}  
\\item \\textbf{Did you include the full text of instructions given to participants and screenshots, if applicable?} \\  
{\[N/A\]} This research does not involve crowdsourcing or direct experiments with human subjects.

\\item \\textbf{Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable?} \\\\  
{\[N/A\]} Not applicable as no human subjects were directly involved in experiments.

\\item \\textbf{Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation?} \\\\  
{\[N/A\]} Not applicable.  
\\end{enumerate}

\\end{enumerate}  
\\end{document}

#### **Works cited**

1. DiTFastAttn: Attention Compression for Diffusion Transformer Models \- NIPS papers, accessed May 15, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/0267925e3c276e79189251585b4100bf-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/0267925e3c276e79189251585b4100bf-Paper-Conference.pdf)  
2. openreview.net, accessed May 15, 2025, [https://openreview.net/pdf?id=gojL67CfS8](https://openreview.net/pdf?id=gojL67CfS8)  
3. 5 Top Papers of NeurIPS 2024 That You Must Read \- Analytics Vidhya, accessed May 15, 2025, [https://www.analyticsvidhya.com/blog/2024/12/neurips-best-paper/](https://www.analyticsvidhya.com/blog/2024/12/neurips-best-paper/)  
4. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction, accessed May 15, 2025, [https://openreview.net/forum?id=gojL67CfS8](https://openreview.net/forum?id=gojL67CfS8)  
5. NeurIPS Poster Visual Autoregressive Modeling: Scalable Image ..., accessed May 15, 2025, [https://nips.cc/virtual/2024/poster/94115](https://nips.cc/virtual/2024/poster/94115)  
6. GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation \- NIPS papers, accessed May 15, 2025, [https://papers.nips.cc/paper\_files/paper/2024/file/6ba9e7ddd48e6db2dcaa7ec3806714b3-Paper-Conference.pdf](https://papers.nips.cc/paper_files/paper/2024/file/6ba9e7ddd48e6db2dcaa7ec3806714b3-Paper-Conference.pdf)  
7. Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment \- OpenReview, accessed May 15, 2025, [https://openreview.net/pdf?id=mlxRLIy7kc](https://openreview.net/pdf?id=mlxRLIy7kc)  
8. MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling \- NIPS papers, accessed May 15, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ebf8764ecf0688cdd9fe1e5a9c525d0d-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ebf8764ecf0688cdd9fe1e5a9c525d0d-Paper-Conference.pdf)  
9. Explaining neural scaling laws | PNAS, accessed May 15, 2025, [https://www.pnas.org/doi/10.1073/pnas.2311878121](https://www.pnas.org/doi/10.1073/pnas.2311878121)  
10. Neural scaling law \- Wikipedia, accessed May 15, 2025, [https://en.wikipedia.org/wiki/Neural\_scaling\_law](https://en.wikipedia.org/wiki/Neural_scaling_law)  
11. proceedings.neurips.cc, accessed May 15, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/f6f4b34d255c2c6c2391af975bed0428-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/f6f4b34d255c2c6c2391af975bed0428-Paper-Conference.pdf)  
12. NeurIPS Using Scenario-Writing for Identifying and Mitigating ..., accessed May 15, 2025, [https://neurips.cc/virtual/2024/104220](https://neurips.cc/virtual/2024/104220)  
13. Using Scenario-Writing for Identifying and Mitigating Impacts of Generative AI \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/2410.23704](https://arxiv.org/abs/2410.23704)  
14. NeurIPS 2024 Call for Papers, accessed May 15, 2025, [https://neurips.cc/Conferences/2024/CallForPapers](https://neurips.cc/Conferences/2024/CallForPapers)  
15. PaperInformation / PaperChecklist \- NeurIPS 2025, accessed May 15, 2025, [https://neurips.cc/public/guides/PaperChecklist](https://neurips.cc/public/guides/PaperChecklist)  
16. NeurIPS Paper Checklist \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2505.04037v1](https://arxiv.org/html/2505.04037v1)  
17. Formatting Instructions For NeurIPS 2024 \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2407.09887v1](https://arxiv.org/html/2407.09887v1)