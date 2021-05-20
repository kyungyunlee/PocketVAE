
## Model architecture          
  
### Note VQ-VAE encoder 
code in `src/models/vqvae.py`

| Layer            | (#filters, kernel size, stride, padding) | Output shape               |
|------------------|------------------------------------------|----------------------------|
| Concat(N+P)      |                                          | (B, 32, 7*2)               |                                                                                                                                                                                                               
| Linear           |                                          | (B, 32, 128)               |
| bi-GRU           |                                          | (B, 32, 128), (B, 32, 128) |
| Concat & Reshape |                                          | (B, 128*2, 32)             |
| Conv1D           | (128, 4, 2, 1)                           | (B, 128, 16)               |
| Conv1D           | (128, 4, 2, 1)                           | (B, 128, 8)                |
| Conv1D           | (8, 1, 1, 0)                             | (B, 8, 8)                  |

Decoder reverses theses operations. 