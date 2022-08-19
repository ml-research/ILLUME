import sys

from magma.model import *

class MultimodalLMA(MultimodalLM):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        config,
        device=None,
    ):
        super().__init__(lm=lm, tokenizer=tokenizer, config=config, device=device)

    @torch.no_grad()
    def probs_of_choices(
        self,
        embeddings: TensorType["b", "s", "d"],
        choices: torch.Tensor,
    ):
        """
        Generates captions for a batch of embeddings.
        """

        was_training = self.training
        self.eval()
        b, s, d = embeddings.shape
        past_key_values = None

        # do sampling
        outputs = self.lm(
            inputs_embeds=embeddings,
            use_cache=True,
            past_key_values=past_key_values,
        )

        logits = outputs.logits[:, -1, :]
        logits = logits[:, choices]
        probs = F.softmax(logits, dim=-1)
        out = probs

        self.train(was_training)
        return out


@torch.no_grad()
def generate(
    self,
    embeddings: TensorType["b", "s", "d"],
    max_steps: int = 100,
    temperature: float = 0.7,
    filter_logits_fn: Callable = top_k,
    filter_threshold: float = 0.9,
    eos_token: int = None,
    decode: bool = True,
    remove_tokens_after_eos: bool = True,
):
    """
    Generates captions for a batch of embeddings.
    """

    # init values
    eos_token = eos_token or self.eos_token
    was_training = self.training
    self.eval()
    b, s, d = embeddings.shape
    past_key_values = None

    # init output with image tokens
    out = torch.zeros((b, s), dtype=torch.int64).to(self.device) + self.image_token

    # do sampling
    for i in range(max_steps):
        if i == 0:
            outputs = self.lm(
                inputs_embeds=embeddings,
                use_cache=True,
                past_key_values=past_key_values,
            )
        else:
            x = out[:, -1:]
            outputs = self.lm(
                input_ids=x, use_cache=True, past_key_values=past_key_values
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # filter / temperature sample
        if filter_logits_fn in {top_k, top_p}:
            filtered_logits = filter_logits_fn(logits, thres=filter_threshold)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
        try:
            sample = torch.multinomial(probs, 1)
        except RuntimeError:
            # nan in probs
            break

        out = torch.cat((out, sample), dim=-1)

        if eos_token is not None and (sample == eos_token).all():
            break

    if decode:
        captions = []
        for b in out:
            if remove_tokens_after_eos:
                # any tokens after and end of sequence token is produced are also set to the eos token
                eos_index = (b == eos_token).nonzero()
                if eos_index.any():
                    b[eos_index[0] :] = eos_token
            b = b.tolist()
            b = [
                i for i in b if (not i == self.image_token) and (not i == eos_token)
            ]
            caption = self.tokenizer.decode(b)
            captions.append(caption)
        out = captions

    self.train(was_training)
    return out


def get_multimodal_model(
        config_path,
        model_dir="models",
        ckpt_path=None,
        tokenizer_name='gpt2',
        lm_ckpt_path=None
):
    from magma.config import MultimodalConfig
    from magma.utils import get_tokenizer
    from magma.transforms import get_transforms

    tokenizer = get_tokenizer(tokenizer_name)
    config = MultimodalConfig.from_yml(config_path)
    print(config.lm_name)
    model = MultimodalLMA(
        lm=get_language_model(config.lm_name, model_dir=model_dir),
        tokenizer=tokenizer,
        config=config,
    )

    transforms = get_transforms(config.image_size, model=model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location=torch.device("cpu"))
        print(f"loading multimodal transformer checkpoint...")
        model.load_state_dict(sd["module"], strict=False)
        print(f"loaded multimodal transformer from checkpoint {ckpt_path}")
    if lm_ckpt_path is not None:
        sd = torch.load(lm_ckpt_path, map_location=torch.device("cpu"))
        print(f"loading language model checkpoint...")
        model.lm.load_state_dict(sd["module"], strict=False)
        print(f"loaded language model from checkpoint {lm_ckpt_path}")
    return model, transforms, tokenizer
