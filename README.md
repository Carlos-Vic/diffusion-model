# Diffusion Models para Geração de Imagens

Implementação manual de um pipeline de geração de imagens com **Stable Diffusion v1-5**, desenvolvida como material de apresentação acadêmica.

Em vez de usar o pipeline pronto (`StableDiffusionPipeline`), cada componente é carregado e conectado separadamente — o objetivo é tornar visível o que acontece por dentro do modelo.

## O que está implementado

- Pipeline manual com os 4 componentes do Stable Diffusion: VAE, U-Net, CLIP Text Encoder e Scheduler
- Classifier-Free Guidance (CFG) com suporte a negative prompt
- Interface interativa com Gradio que mostra a geração da imagem step a step em tempo real

## Componentes

| Componente | Modelo | Função |
|------------|--------|--------|
| VAE | `runwayml/stable-diffusion-v1-5` | Comprime a imagem 512×512 → espaço latente 64×64 |
| U-Net | `runwayml/stable-diffusion-v1-5` | Remove ruído no espaço latente, guiada pelo texto |
| CLIP Text Encoder | `openai/clip-vit-large-patch14` | Transforma o prompt em embedding 77×768 |
| Scheduler | DPMSolverMultistepScheduler | Controla o ritmo do denoising — 20 steps |

## Como executar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nf4I82Ul0uWwd2onbu91iwpszoFuhq-m?usp=sharing)

O notebook foi desenvolvido para rodar no **Google Colab** com GPU T4.

1. Abra `notebook/diffusion_models.ipynb` no Google Colab
2. Ative a GPU: `Runtime > Change runtime type > T4 GPU`
3. Execute as células em ordem
4. A última célula inicia a interface Gradio — é necessário configurar um token do [ngrok](https://ngrok.com) no Colab Secrets com o nome `NGROK_TOKEN`

### Interface Gradio

A última célula do notebook abre uma interface interativa onde é possível:

- Escrever o **prompt** em inglês descrevendo a imagem que deseja gerar
- Escrever o **negative prompt** em inglês com o que deve ser evitado (ex: `blurry, low quality, bad anatomy`)
- Ajustar o número de **steps** de denoising (mais steps = mais qualidade, mais tempo)
- Ajustar o **CFG (Guidance Scale)** — quanto maior, mais o modelo segue o prompt; quanto menor, mais aleatório e criativo
- Definir ou aleatorizar a **seed** para reprodutibilidade

A geração é exibida **passo a passo em tempo real** — é possível acompanhar o reverse process acontecendo visualmente, do ruído puro até a imagem final.

https://github.com/user-attachments/assets/0946681d-08fe-4844-9a7c-ddfd694b18c0

## Parâmetros de geração

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `prompt` | `a futuristic city skyline...` | O que deve ser gerado |
| `negative_prompt` | `daytime, blurry, low quality` | O que deve ser evitado |
| `num_inference_steps` | 20 | Número de steps de denoising |
| `guidance_scale` | 7.5 | Peso do prompt — valores entre 7 e 8.5 são recomendados |
| `seed` | 42 | Garante reprodutibilidade |

## Referências

- [Stable Diffusion with Diffusers — HuggingFace](https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline)
- [How does Stable Diffusion work? — HuggingFace](https://huggingface.co/blog/stable_diffusion)
- [The Illustrated Stable Diffusion — Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Stable Diffusion Negative Prompt — minimaxir](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)
- [GANs vs Diffusion Models — Sapien](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)
