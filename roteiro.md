# Roteiro da Apresentação — Diffusion Models

---

## 1. Como Diffusion Models funcionam

### Ideia central
Um modelo de difusão aprende a gerar imagens resolvendo um problema mais simples: como remover ruído de uma imagem.

### Forward Process — destruindo a imagem
Pegamos uma foto real e adicionamos ruído gaussiano aos poucos, em vários passos, até ela virar estática pura. Isso é feito com milhares de imagens do dataset.

> Imagem original → um pouco de ruído → mais ruído → ... → ruído puro

Esse processo não tem aprendizado — é só matemática fixa.

### Reverse Process — aprendendo a reconstruir
O modelo (uma rede neural chamada U-Net) aprende a fazer o caminho inverso: dado uma imagem levemente ruidosa, ele aprende a prever qual ruído foi adicionado e remove ele.

> Ruído puro → remove um pouco → remove mais → ... → imagem limpa

Ele treina fazendo isso milhões de vezes com imagens diferentes.

### Como gera imagens novas?
Depois de treinado, o modelo recebe ruído puro aleatório e aplica o reverse process — removendo ruído passo a passo até surgir uma imagem coerente que nunca existiu antes.

### Por que funciona?
Porque remover um pouco de ruído de cada vez é uma tarefa simples o suficiente para uma rede neural aprender bem. Fazer isso 50 vezes seguidas resulta em imagens de altíssima qualidade.

---

## 2. Text-to-Image — como o texto entra no processo

### O problema
Um modelo de difusão padrão gera imagens aleatórias — ele não sabe o que você quer. Precisamos de uma forma de guiar a geração a partir de uma descrição em texto.

### A solução: CLIP
O CLIP é um modelo treinado para entender a relação entre texto e imagem. Ele aprende que a frase "um gato laranja" deve estar próxima de fotos de gatos laranjas num espaço matemático.

Quando você digita um prompt, o CLIP transforma esse texto em um vetor numérico — uma representação matemática do que você quer ver.

### Como o texto guia a geração
Esse vetor é passado para a U-Net durante o reverse process. A cada passo de denoising, a U-Net não está só removendo ruído — ela está removendo ruído na direção do que o texto descreveu.

> Prompt → CLIP → vetor de texto → U-Net usa esse vetor em cada step → imagem condicionada ao texto

### Classifier-Free Guidance (guidance_scale)
Na prática, o modelo faz duas previsões a cada step:
- **Com** o prompt → como seria a imagem seguindo o texto
- **Sem** o prompt → como seria uma imagem aleatória

E combina as duas, dando mais ou menos peso ao prompt via o parâmetro `guidance_scale`. Valores altos (ex: 15) prendem mais ao texto mas perdem diversidade. Valores baixos (ex: 1) geram imagens mais criativas mas que ignoram o prompt.

---

## 3. Stable Diffusion por dentro — Latent Diffusion

### O problema do pixel space
Diffusion models tradicionais rodam o denoising diretamente nos pixels. Uma imagem 512×512 tem 786.432 pixels — fazer 50 passos de denoising nesse espaço é lento e exige muita memória.

### A solução: espaço latente
O Stable Diffusion comprime a imagem antes de rodar a difusão. A imagem 512×512 vira uma representação latente de 64×64 — 64 vezes menor. O denoising acontece nesse espaço comprimido, e só no final a imagem é reconstruída no tamanho original.

### Os 3 componentes

**VAE (Autoencoder Variacional)**
Tem duas partes: o encoder comprime a imagem para o espaço latente, e o decoder reconstrói a imagem a partir do latente. No treino usamos o encoder. Na geração usamos só o decoder.

**U-Net**
É a rede neural que aprende a remover ruído no espaço latente. Tem formato de "U": comprime a representação na primeira metade e expande de volta na segunda, com conexões diretas entre as duas para não perder informação. As camadas de cross-attention são onde o texto entra — elas conectam os embeddings do CLIP ao processo de denoising.

**CLIP Text Encoder**
Transforma o prompt em um vetor numérico de tamanho 77×768. O Stable Diffusion não treina o CLIP do zero — usa o modelo já treinado pela OpenAI com os pesos congelados.

### Fluxo completo na inferência
1. Prompt → CLIP → vetor 77×768
2. Ruído aleatório → latente inicial 64×64
3. Loop 50x: U-Net usa latente + vetor de texto para prever e remover ruído
4. Latente final → VAE Decoder → imagem 512×512

---

## 4. Negative Prompting — controlando o que não deve aparecer

> Fonte: https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/

### O que é
Assim como o prompt positivo indica o que você quer ver, o negative prompt indica o que você **não quer** que apareça na imagem. É uma forma direta de controle sobre a geração.

Exemplos práticos:
- Remover elementos indesejados: `trees, green, sky`
- Melhorar qualidade geral: `blurry, pixelated, low quality, ugly`
- Evitar deformações: `bad anatomy, extra limbs, distorted hands`

### Como funciona tecnicamente
Lembre do classifier-free guidance: a cada step, o modelo faz duas previsões — uma com o prompt e outra sem.

O negative prompt **substitui o texto vazio** dessa segunda previsão:

```
# Sem negative prompt:
resultado = previsão_vazia + scale × (previsão_prompt - previsão_vazia)

# Com negative prompt:
resultado = previsão_negativa + scale × (previsão_prompt - previsão_negativa)
```

Em vez de se afastar de "nada", o modelo se afasta de algo **específico**. O negative prompt age como uma âncora de alta dimensão que o processo de difusão evita ativamente.

### Por que ficou tão importante no SD 2.0
O Stable Diffusion 2.0 trocou o text encoder de CLIPText (OpenAI) para OpenCLIP. Com isso, muitos "truques" de prompt que funcionavam antes pararam de funcionar — inclusive usar nomes de artistas como referência de estilo (ex: "Greg Rutkowski" passou a não ter efeito).

O negative prompt se tornou a alternativa: em vez de adicionar palavras mágicas ao prompt positivo, você remove ativamente o que não quer, com resultados mais confiáveis e previsíveis.

### Negative prompt vs. adições ao prompt positivo
Um resultado surpreendente dos experimentos do artigo: adicionar termos de qualidade ao prompt positivo (`hyper-detailed, 8K, realistic`) às vezes **piora** o resultado — pode destruir o estilo original ou desviar a composição.

O negative prompt é mais cirúrgico: você mantém o prompt original intacto e apenas especifica o que deve ser evitado.

### Limitações
O negative prompt não é perfeito — em alguns casos ele remove justamente o que você queria manter. No artigo, ao tentar melhorar a geração do personagem Ugly Sonic com negative prompts de qualidade, o modelo removeu progressivamente os dentes humanos que eram a característica principal do personagem.

Ou seja: o modelo não "entende" a intenção — ele apenas se afasta matematicamente do espaço definido pelo negative prompt.

---

## 5. Demonstração prática — o pipeline no código

### Visão geral
Ao invés de usar o pipeline pronto da biblioteca, implementamos os componentes separadamente para mostrar exatamente o que acontece por dentro.

### Carregando os componentes
```python
vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
scheduler    = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
```
Cada linha carrega um dos componentes separadamente. Isso deixa claro que o Stable Diffusion não é um modelo monolítico — é uma composição de peças independentes.

### Codificando o texto
```python
text_input      = tokenizer(prompt, ...)
text_embeddings = text_encoder(text_input.input_ids)[0]

neg_input      = tokenizer(negative_prompt, ...)
neg_embeddings = text_encoder(neg_input.input_ids)[0]

text_embeddings = torch.cat([neg_embeddings, text_embeddings])
```
O tokenizer transforma o texto em tokens numéricos. O text encoder (CLIP) transforma esses tokens em embeddings. Concatenamos o negativo e o positivo para fazer os dois passes do classifier-free guidance de uma vez só.

### O ruído inicial
```python
latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8))
latents = latents * scheduler.init_noise_sigma
```
`torch.randn` gera ruído gaussiano puro. O shape `(1, 4, 64, 64)` confirma que estamos no espaço latente — não em pixels. Multiplicamos por `init_noise_sigma` para escalar o ruído na magnitude que o scheduler espera.

### O loop de denoising
```python
for t in tqdm(scheduler.timesteps):
    latent_input = torch.cat([latents] * 2)
    noise_pred   = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

    noise_neg, noise_pos = noise_pred.chunk(2)
    noise_pred = noise_neg + guidance_scale * (noise_pos - noise_neg)

    latents = scheduler.step(noise_pred, t, latents).prev_sample
```
A cada step, a U-Net recebe o latente ruidoso e os embeddings do texto e prevê qual ruído remover. O classifier-free guidance combina a previsão negativa e positiva. O scheduler aplica a remoção e retorna o latente um pouco menos ruidoso.

### Decodificando para imagem
```python
latents = 1 / 0.18215 * latents
image   = vae.decode(latents).sample
```
O fator `0.18215` é uma constante de normalização do VAE do Stable Diffusion. Depois disso, o decoder reconstrói a imagem 512×512 a partir do latente 64×64.

---

## 6. Diffusion Models vs GANs

> Fonte: https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis

### O que são GANs?
GANs (Generative Adversarial Networks) foram introduzidas por Ian Goodfellow em 2014. Funcionam com dois componentes que competem entre si:

- **Generator:** gera imagens falsas a partir de ruído aleatório
- **Discriminator:** tenta distinguir imagens reais das geradas

Esse "jogo" faz o generator melhorar continuamente até produzir imagens indistinguíveis das reais.

### Como os dois se comparam

| Critério | GANs | Diffusion Models |
|----------|------|-----------------|
| Velocidade de geração | Rápida — uma passagem pela rede | Lenta — 20 a 50 steps iterativos |
| Qualidade da imagem | Alta | Alta ou superior |
| Diversidade das amostras | Baixa — risco de mode collapse | Alta — cobre melhor a distribuição |
| Estabilidade do treino | Instável — requer ajuste fino | Estável — objetivo de treino mais simples |
| Custo computacional | Menor na inferência | Maior na inferência |

### Mode Collapse — o grande problema dos GANs
O generator pode "travar" gerando sempre as mesmas amostras que enganam o discriminator, ignorando toda a diversidade do dataset. Diffusion models não sofrem desse problema porque o objetivo de treino é remover ruído — não enganar ninguém.

### Quando usar cada um?
- **GANs:** quando velocidade é prioridade — tempo real, jogos, aplicações interativas
- **Diffusion Models:** quando qualidade e diversidade são prioridade — geração de imagens, síntese de áudio, dados científicos

### Por que os Diffusion Models dominaram a geração de imagens?
O paper de Dhariwal & Nichol (2021) — *"Diffusion Models Beat GANs on Image Synthesis"* — mostrou empiricamente que diffusion models superam GANs em qualidade de imagem sem sofrer de mode collapse. A partir daí, ferramentas como Stable Diffusion, DALL-E 2 e Midjourney adotaram a abordagem de difusão.

---

## 7. Discussão Crítica

### Direitos Autorais
Os modelos são treinados em bilhões de imagens coletadas da internet, sem consentimento dos artistas originais. A imagem gerada pode replicar estilos ou elementos de obras protegidas.

**O caso Getty Images vs. Stability AI**
A Getty Images processou a Stability AI em diversas jurisdições — incluindo Estados Unidos e Reino Unido — alegando uso indevido de milhões de suas fotografias para treinar o Stable Diffusion sem autorização.

Em novembro de 2025, a Alta Corte de Justiça do Reino Unido proferiu a primeira decisão sobre o caso, com resultado dividido:

- **Vitória parcial da Getty Images:** a Stability AI foi considerada responsável por gerar inadvertidamente marcas d'água da Getty em versões antigas do modelo, configurando violação de marca registrada.
- **Vitória da Stability AI nos direitos autorais:** o tribunal concluiu que o modelo de IA não constitui uma "cópia infratora" das obras usadas no treinamento — pois ele aprende padrões, mas não armazena nem reproduz as imagens originais.

O precedente é importante: segundo a legislação britânica vigente, um modelo de IA não é ilegalmente uma cópia das obras que o treinaram. A decisão ainda é passível de recurso.

**Pergunta em aberto:** quem é o autor de uma imagem gerada por IA? O usuário que escreveu o prompt? Os artistas cujo trabalho treinou o modelo? A empresa que desenvolveu a ferramenta? A legislação ainda não tem resposta consolidada.

### Deepfakes
A facilidade de gerar rostos e cenas hiper-realistas criou riscos sérios:
- Desinformação: vídeos e imagens falsas de figuras públicas
- Fraude: identidades sintéticas para golpes financeiros
- Abuso de imagem: criação de conteúdo não consensual

**Respostas técnicas atuais:** watermarking invisível nas imagens geradas e classificadores treinados para detectar conteúdo sintético.

### Consumo Computacional
Treinar o Stable Diffusion 1.x custou aproximadamente $600.000 e consumiu ~600 MWh de energia — equivalente ao consumo anual de dezenas de residências.

A inferência é barata, mas o treino do zero está fora do alcance da maioria. Isso concentra o poder de desenvolvimento em poucas empresas com recursos computacionais massivos.

**Tendência:** modelos menores e mais eficientes via destilação (SDXL-Turbo, LCM) buscam reduzir esse custo.

### Controle da Geração
Quanto controle o usuário deve ter sobre o que o modelo gera? As ferramentas atuais respondem de formas diferentes:
- Safety checkers que bloqueiam conteúdo NSFW
- Negative prompts como mecanismo de controle pelo usuário
- Modelos open-source sem restrições vs. APIs fechadas com filtros rígidos

Não existe consenso — é um debate técnico, ético e regulatório ainda em aberto.

---

## 8. Tópicos Avançados e Estado da Arte

As arquiteturas baseadas em difusão evoluíram consideravelmente desde sua formulação inicial. Os avanços recentes concentram-se na redução do custo computacional, no refinamento do controle de geração e no desenvolvimento de novas abordagens de segurança.

### 8.1. Evolução Arquitetural: Diffusion Transformers (DiT)
Tradicionalmente, os processos de difusão dependiam da arquitetura convolucional U-Net. Com o desenvolvimento dos **Diffusion Transformers (DiT)**, demonstrou-se que a adoção de convoluções não é estritamente necessária. Os DiTs aproveitam as leis de escalabilidade (*scaling laws*) inerentes aos Transformers, otimizando o processamento em grandes volumes de dados.

**Estrutura e Condicionamento**
Na arquitetura DiT, o espaço latente é fragmentado em *patches* convertidos em tokens. O condicionamento (via texto ou variáveis de ruído) é aplicado utilizando blocos de *Adaptive Layer Normalization* (adaLN). O modelo é inicializado como função de identidade para garantir estabilidade computacional nas etapas iniciais de treinamento.

**Eficiência Computacional**
A manipulação da resolução latente afeta de maneira previsível o balanço entre carga computacional (Gflops) e a qualidade da geração (FID).

| Arquitetura | Tamanho do Patch Latente | Resolução | FID (ImageNet) | Gflops | Contagem de Parâmetros |
|-------------|--------------------------|-----------|----------------|--------|------------------------|
| DiT-XL/8    | 8                        | 256x256   | Alta (Menos detalhes) | Baixo  | 675M                   |
| DiT-XL/2    | 2                        | 256x256   | 2.27           | 119    | 675M                   |
| DiT-XL/2    | 2                        | 512x512   | 3.04           | 525    | 675M                   |
| ADM-U (U-Net)| N/A                     | 512x512   | 3.85           | 2813   | >1B                    |

*(Tabela: Comparativo de desempenho entre DiT e U-Net em relação à eficiência e qualidade)*

### 8.2. Multimodalidade e MMDiT
Em versões anteriores, a renderização de tipografia falhava devido à restrição da atenção cruzada unidirecional. A implementação do **Multimodal Diffusion Transformer (MMDiT)**, adotada em modelos como Stable Diffusion 3 e FLUX, permite a interação bidirecional entre tokens de imagem e texto, viabilizando resultados tipográficos consistentes.

### 8.3. Otimização de Processamento: Flow Matching e Rectified Flows
Modelos baseados em SDEs (Equações Diferenciais Estocásticas) tendem a demandar um grande número de iterações de remoção de ruído. A técnica de **Flow Matching** otimiza o trajeto de geração por meio do aprendizado de um campo de velocidade contínuo em ODEs. Quando combinada com **Rectified Flows**, a técnica aproxima as trajetórias matemáticas de um comportamento linear, reduzindo a quantidade de passos necessários para a síntese qualitativa.

### 8.4. Modelos de Consistência (Consistency Models)
A abordagem de **Consistency Models** altera o paradigma de geração iterativa ao mapear diretamente pontos ao longo da trajetória de difusão para a origem não ruidosa. Treinados via destilação de um modelo primário ou de modo independente, estes modelos alcançam a capacidade de gerar imagens em um passo computacional unificado, otimizando o tempo de latência em relação à arquitetura convencional.

### 8.5. Condicionamento Visual Estrutural: IP-Adapters
O **IP-Adapter** consolidou-se como um mecanismo de indução de estilo e estrutura visual. Sem a necessidade de alterar os parâmetros da rede original, a técnica integra redes adicionais que operam por meio de atenção cruzada desacoplada. Desta forma, atributos de uma imagem de referência influenciam a geração de maneira concomitante ao controle semântico estipulado via texto.

### 8.6. Aplicação Tridimensional e Processamento de Vídeo
Na transição para dados tridimensionais, técnicas como a **Score Distillation Sampling (SDS)** projetam distribuições 2D em representações espaciais (e.g., *3D Gaussian Splatting*). No contexto de vídeos, arquiteturas baseadas em DiT interpretam dados em formato de blocos latentes espaço-temporais (*spacetime latent patches*), capacitando o modelo a estipular fluidez e consistência cronológica em quadros dinâmicos.

### 8.7. Mecanismos Adversariais e Autenticidade
- **Proteção e Engenharia Adversarial:** Plataformas defensivas, como Glaze e Nightshade, permitem aos criadores aplicar perturbações adversariais em imagens para inibir processos de extração de padrões. Simultaneamente, pesquisas independentes formulam contramedidas heurísticas (como o sistema LightShed), cujo objetivo é detectar e neutralizar essas perturbações em tempo de processamento.
- **Identificação Digital:** Para rastreamento de autoria, soluções tecnológicas como o SynthID inserem marcas criptográficas diretamente no domínio numérico da matriz da imagem. Este método propõe garantir a identificação forense inalterada mesmo perante processamentos severos de compressão, recorte e filtragem.

---

## Fontes

| Seção | Fonte |
|-------|-------|
| 1. Como Diffusion Models funcionam | https://www.ibm.com/br-pt/think/topics/diffusion-models |
| 3 e 5. Stable Diffusion e Pipeline | https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline |
| 4. Negative Prompting | https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/ |
| 6. GANs vs Diffusion Models | https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis |
| 7. Direitos Autorais — decisão britânica | https://ids.org.br/noticia/getty-images-vs-stability-ai-tribunal-britanico-afirma-que-um-modelo-de-inteligencia-artificial-generativa-nao-e-por-si-so-uma-copia-infratora/ |
| 7. Direitos Autorais — processo Getty | https://exame.com/inteligencia-artificial/getty-images-vai-a-justica-contra-stability-ai-por-uso-indevido-de-imagens/ |

---

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0.0 | Criação do roteiro base | | |  |  |
| 1.1.0 | Adição de Tópicos Avançados (DiT, LCM, IP-Adapter) | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 24/06/2026 |  | 24/06/2026 |
