# Paligemma-MultiModal-System
Paligemma Multi-Modal System: A Comprehensive implementation from Scratch, Emulating Multi-Modal Architectures for Future Inference and Deployment

### Features
- Contrastive Vision Model
- SigLIP vs CLIP
- RoPE (`Rotary Positional Encoding`)
- MHA (`Multi-head attention`)
- GQA (`Grouped Query Attention`)
- Normalisation Layers
- KV-Cache (`Key-Value Cache`)
- Contrastive Vision Model ( pre-filling and token-generation )
- Weight Tying (`Memory Optimisation technique`)
- Gemma String (`Specific to the paper during preprocessing`)
- Top-p sampling 
- Temperature
- PEFT (`Fine-tuning optimisations`)
    - LoRA
    - Quantisation
- Gradient Checkpointing (`Memory Optimisations incase of Out-of-Memory Error`)
- Multi-gpu training (`Production Setup`)


### The overall Multi-Modal Architecture [Abstraction]
![Alt text](./design_diagrams/design_images/Multi_Modal_System.drawio.png)


### Block-Subblock Dependency View [Concrete]
![Alt text](./design_diagrams/design_images/Multi_Modal_System_white_mod.drawio.png)
