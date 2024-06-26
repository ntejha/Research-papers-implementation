# GPT Architecture

## Transformer Architecture

First off, we have the inputs and outputs (shifted right) going through the embeddings (with positional encoding). All this passes through a number of layers. Let's take 4 layers as an example. The inputs will go through 4 encoders, and then the outputs and final encoder values pass through the 4 decoders. Finally, the last decoder will perform a transformation, pass through the softmax layer, and produce the probability distribution for token generation.

Second, each encoder consists of four main processes:

1. **Multi-head Attention**
   - There are multiple heads learning different semantic information from unique perspectives, with each head having its own learnable parameters. Let's delve into the inner workings:
     - **Keys, Query, Value**: Essentially, the key represents all the words in a sentence, the query represents the word we are focusing on, and their dot product gives the attention score.
     - **Scaled Dot-product Attention (n heads running in parallel)**:
       - First, the key and query are dot-multiplied to form the attention scores. These values are scaled to avoid excessively large values. Sometimes, we apply torch.tril masking to prevent looking at future tokens. Then, we apply softmax to get more confident attention scores, and finally, we multiply the matrix with the value (which is the output of the input vectors with attention applied to each token).
     - **Concatenate Results**
     - **Linear Layer**

2. **Residual Connection (Add) then Normalize**

3. **Feed Forward Network (Linear -> ReLU -> Linear)**

4. **Residual Connection (Add) then Normalize**

These processes feed into the next encoder block. If it's the last encoder block, it feeds into each decoder block.

- **Multi-head Attention**: Looks at past, present, and future tokens.
- **Masked Multi-head Attention**: Looks at past and present tokens only.

## Self-Attention Working

Let's take a four-token sequence "My dog has fleas". We'll highlight which words correlate with each other, showing how the attention mechanism multiplies them together to get high scores. This demonstrates how GPT understands internally.

```
        My    dog   has   fleas
My      low   med   low    low
dog     med   low   med    high
has     low   med   low    med
fleas   low   high  med    low
```

From these values, you can see which tokens have high and low attention scores, helping the network learn to place attention scores on the tokens through embeddings. Attention is used to generate tokens, which is the core of how GPT works.

## GPT Architecture

GPT operates similarly to the Transformer architecture, but the encoder part is removed. Additionally, scaling is used in attention scores to prevent the vanishing gradient problem.

### GPT Process:

- Tokenized inputs
- Embeddings with positional encoding
- n Decoders according to the layers

  In each decoder, the process is:
  - **Multi-head Attention**: Multiple heads run in parallel. Each head involves:
    - **Keys, Query, and Value**
    - **Scaled Dot-product Attention**: For each head, the key and query are dot-multiplied, then scaled to combat the vanishing gradient. Torch.tril masking is applied to prevent looking into the future, followed by softmax. The resulting values are matrix-multiplied with the value, producing a blend of input vectors and attention on each token.
    - **Concatenate Results**
    - **Linear Layer**
  - **Residual Connection (Add) then Normalize**
  - **Feed Forward Network (Linear -> ReLU -> Linear)**
  - **Residual Connection (Add) then Normalize**

- Linear Layer
- Softmax Layer
- Probability Sampling Generation
- Compare Targets to Outputs -> Backpropagate

The dataset used for this architecture is [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/).

Thanks to Elliotcodes for giving me a clear insight on how this architecture worked.
